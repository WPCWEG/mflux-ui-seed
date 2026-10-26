import gc
import io
import json
import logging
import queue
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from mflux.web.seed.adapters import AdapterError, ModelCacheKey
from mflux.web.seed.invocation import Invocation, InvocationError

logger = logging.getLogger(__name__)


class JobCancelled(Exception): ...


@dataclass
class Job:
    invocation: Invocation
    payload: dict
    preview_every: int = 0
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    images: list[str] = field(default_factory=list)
    error: str | None = None
    messages: list[str] = field(default_factory=list)
    preview_jpeg: bytes | None = None
    events: list[dict] = field(default_factory=list)
    tiling_config: dict = field(default_factory=dict)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    TERMINAL = ("done", "error", "cancelled")

    def emit(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append({"seq": len(self.events), "type": kind, "time": time.time(), **data})

    def events_since(self, seq: int) -> list[dict]:
        with self._lock:
            return self.events[seq:]

    @property
    def finished(self) -> bool:
        return self.status in Job.TERMINAL

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "command": self.invocation.command,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "images": list(self.images),
                "error": self.error,
                "messages": list(self.messages),
                "shell_command": self.invocation.shell_command(),
                "prompt": (self.payload.get("options") or {}).get("--prompt", ""),
                "tiling": dict(self.tiling_config),
            }


class WebProgressCallback:
    # Registered once per cached model; forwards to whichever job the runner is executing.
    def __init__(self, runner: "JobRunner", model, latent_creator):
        self.runner = runner
        self.model = model
        self.latent_creator = latent_creator

    def call_before_loop(self, seed, prompt, latents, config, **kwargs) -> None:
        job = self.runner.current_job
        if job is None:
            return
        battery_saver = self.runner.battery_saver
        if battery_saver is not None:
            battery_saver.call_before_loop()
        job.emit("seed_start", seed=seed, total=config.num_inference_steps, start=config.init_time_step)

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:
        job = self.runner.current_job
        if job is None:
            return
        if job.cancel_requested.is_set():
            from mflux.utils.exceptions import StopImageGenerationException

            raise StopImageGenerationException(f"Cancelled at step {t + 1}/{config.num_inference_steps}")
        info = time_steps.format_dict if time_steps is not None else {}
        job.emit(
            "progress",
            seed=seed,
            step=t + 1,
            total=config.num_inference_steps,
            elapsed=info.get("elapsed"),
            rate=info.get("rate"),
        )
        if job.preview_every and (t + 1) % job.preview_every == 0 and t + 1 < config.num_inference_steps:
            try:
                job.preview_jpeg = self._preview(latents, config, seed, prompt)
                job.emit("preview", seed=seed, step=t + 1)
            except Exception as exc:  # noqa: BLE001 -- a failed preview must not kill the run
                logger.warning("Preview decode failed: %s", exc)
                job.preview_every = 0

    def call_after_loop(self, seed, prompt, latents, config) -> None:
        job = self.runner.current_job
        if job is not None:
            job.emit("decoding", seed=seed)

    def _preview(self, latents, config, seed, prompt) -> bytes:
        # Same decode path as StepwiseHandler, downscaled to a JPEG for the browser.
        from mflux.utils.image_util import ImageUtil

        unpacked = self.latent_creator.unpack_latents(latents=latents, height=config.height, width=config.width)
        vae = self.model.vae
        if hasattr(vae, "decode_packed_latents") and unpacked.shape[1] > getattr(vae, "latent_channels", 32):
            decoded = vae.decode_packed_latents(unpacked)
        else:
            decoded = vae.decode(unpacked)
        image = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.model.bits,
            generation_time=0,
        ).image
        image.thumbnail((512, 512))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()


class JobRunner:
    MAX_JOBS_KEPT = 200
    # Queue sentinel: model memory may only be touched on the worker thread that owns MLX.
    UNLOAD = "unload"
    GIB = 1024**3
    GB = GIB
    DEFAULT_MEMORY_LIMIT_BYTES = 16 * GIB  # 17,179,869,184 B (16 GiB guideline)
    DEFAULT_CACHE_LIMIT_BYTES = 1 * GIB  # 1,073,741,824 B (1 GiB idle cache limit)

    def __init__(
        self,
        cache_size: int = 1,
        idle_unload_seconds: float = 600,
        baseline_memory_limit: int | None = DEFAULT_MEMORY_LIMIT_BYTES,
        baseline_cache_limit: int | None = DEFAULT_CACHE_LIMIT_BYTES,
    ):
        self.cache_size = max(cache_size, 0)
        self.idle_unload_seconds = max(idle_unload_seconds, 0)
        self.baseline_memory_limit = baseline_memory_limit
        self.baseline_cache_limit = baseline_cache_limit
        self.current_job: Job | None = None
        self.battery_saver = None
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: "queue.Queue[Job | str | None]" = queue.Queue()
        self._models: OrderedDict[tuple, object] = OrderedDict()
        self._jobs_lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, name="mflux-web-worker", daemon=True)
        self._loading: str | None = None
        self._last_used = time.monotonic()
        self._initial_system_memory_limit: int | None = None
        self._initial_system_cache_limit: int | None = None
        self._current_memory_limit: int | None = baseline_memory_limit
        self._current_cache_limit: int | None = baseline_cache_limit
        self._stopping: bool = False

    @property
    def stopping(self) -> bool:
        return self._stopping

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> bool:
        self._stopping = True
        with self._jobs_lock:
            for job in self._jobs.values():
                if job.status == "queued":
                    with job._lock:
                        job.cancel_requested.set()
                        job.status = "cancelled"
                        job.finished_at = time.time()
                        job.error = "Server stopping"
                        job.emit("cancelled", message="Server stopping")
        current = self.current_job
        if current is not None:
            current.cancel_requested.set()
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        exited = not self._thread.is_alive()
        if not exited:
            logger.warning("Worker thread did not stop within %.1fs timeout", timeout)
        else:
            logger.info("Worker thread stopped cleanly")
        return exited

    def submit(self, invocation: Invocation, payload: dict, preview_every: int = 0) -> Job:
        job = Job(invocation=invocation, payload=payload, preview_every=max(int(preview_every or 0), 0))
        with self._jobs_lock:
            if self._stopping:
                raise InvocationError("Server is stopping; no new jobs accepted")
            self._jobs[job.id] = job
            while len(self._jobs) > JobRunner.MAX_JOBS_KEPT:
                oldest_id, oldest = next(iter(self._jobs.items()))
                if not oldest.finished:
                    break
                self._jobs.pop(oldest_id)
            waiting = [j for j in self._jobs.values() if j.status == "queued"]
            position = waiting.index(job) + 1 if job in waiting else 0
            job.emit("queued", position=position)
            self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        with self._jobs_lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.finished:
            return False
        with job._lock:
            if job.finished:
                return False
            job.cancel_requested.set()
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = time.time()
                job.error = "Removed from queue"
                job.emit("cancelled", message="Removed from queue")
                return True
        return True

    def clear_history(self) -> int:
        # Finished jobs only: running and queued ones still have a worker or client attached.
        with self._jobs_lock:
            finished = [job_id for job_id, job in self._jobs.items() if job.finished]
            for job_id in finished:
                self._jobs.pop(job_id)
        return len(finished)

    def active_uploads(self) -> set[str]:
        uploads = set()
        for job in self.jobs():
            image = (job.payload or {}).get("image") or {}
            if not job.finished and image.get("upload"):
                uploads.add(str(image["upload"]))
        return uploads

    def request_unload(self) -> bool:
        if not self._models:
            return False
        self._queue.put(JobRunner.UNLOAD)
        return True

    def queue_position(self, job: Job) -> int:
        waiting = [j for j in self.jobs() if j.status == "queued"]
        return waiting.index(job) + 1 if job in waiting else 0

    def status(self) -> dict:
        return {
            "current_job": self.current_job.id if self.current_job else None,
            "queued": sum(1 for j in self.jobs() if j.status == "queued"),
            "loading": self._loading,
            "cached_models": [ModelCacheKey.describe(key) for key in list(self._models)],
            "cache_size": self.cache_size,
            "memory": JobRunner.memory_snapshot(),
            "limits": self.limits_snapshot(),
            "unload_in": self._seconds_until_idle_unload(),
            "stopping": self.stopping,
        }

    def _set_memory_limit(self, limit: int) -> None:
        import mlx.core as mx

        mx.set_memory_limit(limit)
        self._current_memory_limit = limit

    def _set_cache_limit(self, limit: int) -> None:
        import mlx.core as mx

        mx.set_cache_limit(limit)
        self._current_cache_limit = limit

    def limits_snapshot(self) -> dict:
        cur_mem = self._current_memory_limit or 0
        cur_cache = self._current_cache_limit or 0
        base_mem = self.baseline_memory_limit
        base_cache = self.baseline_cache_limit
        return {
            "memory_limit_bytes": cur_mem,
            "cache_limit_bytes": cur_cache,
            "memory_limit_gib": round(cur_mem / JobRunner.GIB, 2),
            "cache_limit_gib": round(cur_cache / JobRunner.GIB, 2),
            "baseline_memory_limit_bytes": base_mem,
            "baseline_cache_limit_bytes": base_cache,
            "baseline_memory_limit_gib": round(base_mem / JobRunner.GIB, 2) if base_mem is not None else None,
            "baseline_cache_limit_gib": round(base_cache / JobRunner.GIB, 2) if base_cache is not None else None,
        }

    @staticmethod
    def memory_snapshot() -> dict:
        import mlx.core as mx

        return {
            "active_gb": round(mx.get_active_memory() / JobRunner.GIB, 2),
            "cache_gb": round(mx.get_cache_memory() / JobRunner.GIB, 2),
            "peak_gb": round(mx.get_peak_memory() / JobRunner.GIB, 2),
        }

    def _work(self) -> None:
        import mlx.core as mx

        # MLX has no getter for memory and cache limits; read them once on the worker thread.
        self._initial_system_memory_limit = mx.set_memory_limit(0)
        self._set_memory_limit(self._initial_system_memory_limit)
        self._initial_system_cache_limit = mx.set_cache_limit(0)
        self._set_cache_limit(self._initial_system_cache_limit)

        # Apply Web baseline limits: 16 GiB memory guideline, 1 GiB idle cache limit
        if self.baseline_memory_limit is not None:
            self._set_memory_limit(self.baseline_memory_limit)
        if self.baseline_cache_limit is not None:
            self._set_cache_limit(self.baseline_cache_limit)

        try:
            while True:
                try:
                    job = self._queue.get(timeout=self._seconds_until_idle_unload())
                except queue.Empty:
                    self._unload_all(f"idle for {self.idle_unload_seconds / 60:g} min")
                    continue
                if job is None:
                    return
                if isinstance(job, str) and job == JobRunner.UNLOAD:
                    self._unload_all("unload requested")
                    continue
                if job.cancel_requested.is_set():
                    continue
                with job._lock:
                    if job.cancel_requested.is_set() or job.finished:
                        continue
                    job.status = "running"
                    job.started_at = time.time()
                mx.reset_peak_memory()
                self.current_job = job
                job.emit("started")
                try:
                    self._run(job)
                    with job._lock:
                        job.finished_at = time.time()
                        job.status = "done"
                        job.emit("done", images=job.images)
                except JobCancelled as exc:
                    with job._lock:
                        job.finished_at = time.time()
                        job.status = "cancelled"
                        job.error = str(exc)
                        job.emit("cancelled", message=str(exc), images=job.images)
                except Exception as exc:  # noqa: BLE001 -- every failure is reported on the job
                    error_msg = JobRunner._describe_error(exc)
                    if not isinstance(exc, (InvocationError, AdapterError)):
                        logger.exception("Job %s failed", job.id)
                        self._unload_all(f"evicted after unexpected error: {error_msg}")
                    with job._lock:
                        job.finished_at = time.time()
                        job.status = "error"
                        job.error = error_msg
                        job.emit("error", message=error_msg, images=job.images)
                finally:
                    if job.finished_at is None:
                        job.finished_at = time.time()
                    self.current_job = None
                    self.battery_saver = None
                    self._after_job(job)
        finally:
            self._unload_all("worker stopping")
            if self._initial_system_memory_limit is not None:
                self._set_memory_limit(self._initial_system_memory_limit)
            if self._initial_system_cache_limit is not None:
                self._set_cache_limit(self._initial_system_cache_limit)
            logger.info("Worker thread exited; all models unloaded and limits restored")

    def _after_job(self, job: Job) -> None:
        import mlx.core as mx

        # Restore baseline limits after every job (done, cancelled, or error)
        if self.baseline_memory_limit is not None:
            self._set_memory_limit(self.baseline_memory_limit)
        if self.baseline_cache_limit is not None:
            self._set_cache_limit(self.baseline_cache_limit)
        peak_gb = mx.get_peak_memory() / JobRunner.GIB
        gc.collect()
        mx.clear_cache()
        self._last_used = time.monotonic()
        memory = JobRunner.memory_snapshot()
        loaded = ", ".join(ModelCacheKey.describe(key)["model"] or key[0] for key in self._models) or "none"
        logger.info(
            "Job %s %s in %.1fs · peak %.1f GiB · now %.1f GiB active (%.1f GiB cached) · models loaded: %s",
            job.id,
            job.status,
            (job.finished_at or time.time()) - (job.started_at or job.created_at),
            peak_gb,
            memory["active_gb"],
            memory["cache_gb"],
            loaded,
        )

    def _seconds_until_idle_unload(self) -> float | None:
        if not self._models or not self.idle_unload_seconds or self.current_job is not None:
            return None
        return max(self._last_used + self.idle_unload_seconds - time.monotonic(), 0.01)

    def _unload_all(self, reason: str) -> None:
        if not self._models:
            return
        names = ", ".join(ModelCacheKey.describe(key)["model"] or key[0] for key in self._models)
        self._models.clear()
        JobRunner._release_memory()
        logger.info("Unloaded %s (%s) · now %.1f GiB active", names, reason, JobRunner.memory_snapshot()["active_gb"])

    def _run(self, job: Job) -> None:
        from mflux.callbacks.instances.battery_saver import BatterySaver
        from mflux.utils.exceptions import StopImageGenerationException
        from mflux.utils.prompt_util import PromptUtil

        invocation = job.invocation
        adapter = invocation.adapter
        args = invocation.parse()
        adapter.validate(args)
        job.messages.extend(args.web_messages)
        if job.cancel_requested.is_set():
            raise JobCancelled("Cancelled")
        if args.mlx_cache_limit_gb is not None:
            max_cache_gib = (self.baseline_memory_limit / JobRunner.GIB) if self.baseline_memory_limit else 16.0
            if args.mlx_cache_limit_gb > max_cache_gib:
                raise InvocationError(
                    f"--mlx-cache-limit-gb ({args.mlx_cache_limit_gb:g} GiB) exceeds the host memory guideline of {max_cache_gib:g} GiB"
                )
            self._set_cache_limit(int(args.mlx_cache_limit_gb * JobRunner.GIB))
        self.battery_saver = BatterySaver(battery_percentage_stop_limit=args.battery_percentage_stop_limit)

        if job.cancel_requested.is_set():
            raise JobCancelled("Cancelled")

        cacheable = not args.low_ram and self.cache_size > 0
        model = self._cached_model(job, args) if cacheable else self._uncached_model(job, args)
        self._apply_tiling(model, args, job=job)
        try:
            for index, seed in enumerate(args.seed):
                if job.cancel_requested.is_set():
                    raise JobCancelled("Cancelled")
                job.emit("seed_queued", seed=seed, index=index, count=len(args.seed))
                try:
                    image = adapter.generate(model, args, seed, PromptUtil.read_prompt(args))
                except StopImageGenerationException as exc:
                    raise JobCancelled(str(exc)) from exc
                job.images.append(self._save(job, image, args, seed))
                job.emit("image", name=job.images[-1], seed=seed)
        finally:
            if not cacheable:
                del model
                JobRunner._release_memory()

    def _cached_model(self, job: Job, args):
        key = ModelCacheKey.of(job.invocation.command, args)
        if key in self._models:
            self._models.move_to_end(key)
            job.emit("model_cached", model=ModelCacheKey.describe(key))
            return self._models[key]["model"]
        while len(self._models) >= self.cache_size:
            _, evicted = self._models.popitem(last=False)
            del evicted
            JobRunner._release_memory()
        model = self._load(job, args, key)
        self._models[key] = {"model": model, "tiling": getattr(model, "tiling_config", None)}
        return model

    def _uncached_model(self, job: Job, args):
        from mflux.callbacks.callback_manager import CallbackManager

        # Low-RAM runs use the CLI's own MemorySaver, which deletes encoders and transformer
        # as it goes, so the model cannot be reused and the cache is emptied first.
        self._models.clear()
        JobRunner._release_memory()
        key = ModelCacheKey.of(job.invocation.command, args)
        model = self._load(job, args, key)

        # Apply tiling before callbacks so MemorySaver knows if tiling was explicitly disabled:
        self._apply_tiling(model, args, job=job)

        CallbackManager.register_callbacks(
            args=args, model=model, latent_creator=job.invocation.adapter.latent_creator()
        )

        # Re-enforce tiling policy in case Low RAM MemorySaver implicitly installed one:
        self._apply_tiling(model, args, job=job)
        return model

    def _load(self, job: Job, args, key: tuple):
        described = ModelCacheKey.describe(key)
        job.emit("loading", model=described)
        self._loading = described["model"] or job.invocation.command
        started = time.time()
        try:
            model = job.invocation.adapter.load(args)
        finally:
            self._loading = None
        model.callbacks.register(WebProgressCallback(self, model, job.invocation.adapter.latent_creator()))
        job.emit("loaded", model=described, seconds=round(time.time() - started, 1))
        return model

    def _apply_tiling(self, model, args, job: Job | None = None) -> None:
        from mflux.models.common.vae.tiling_config import TilingConfig

        entry = next((e for e in self._models.values() if e["model"] is model), None)
        raw_options = getattr(args, "web_raw_options", {})
        explicitly_disabled = raw_options.get("--vae-tiling") is False

        cmd = (
            getattr(args, "web_command", "")
            or getattr(args, "command", "")
            or (job.invocation.command if job is not None else "")
        )
        is_z_image = cmd.startswith("mflux-generate-z-image") or (
            hasattr(model, "__class__") and "ZImage" in model.__class__.__name__
        )

        tile_size = getattr(args, "vae_tile_size", None)
        wants_tiling = bool(
            (getattr(args, "vae_tiling", False) or tile_size is not None or (is_z_image and not explicitly_disabled))
            and not explicitly_disabled
        )

        if hasattr(model, "vae") and not hasattr(model.vae, "_web_initial_implicit_tiling"):
            model.vae._web_initial_implicit_tiling = getattr(model.vae, "supports_implicit_tiling", False)

        if wants_tiling:
            effective_size = tile_size or 512
            model.tiling_config = TilingConfig(vae_decode_tile_size=effective_size, vae_decode_overlap=8)
            if hasattr(model, "vae") and hasattr(model.vae, "_web_initial_implicit_tiling"):
                model.vae.supports_implicit_tiling = model.vae._web_initial_implicit_tiling
            effective_desc = f"enabled (size={effective_size}, overlap=8)"
        else:
            model.tiling_config = None
            if hasattr(model, "vae"):
                model.vae.supports_implicit_tiling = False
            effective_desc = "explicitly disabled" if explicitly_disabled else "disabled"

        if entry is not None:
            entry["tiling"] = model.tiling_config

        if job is not None:
            job.tiling_config = (
                {
                    "enabled": True,
                    "tile_size": model.tiling_config.vae_decode_tile_size,
                    "overlap": model.tiling_config.vae_decode_overlap,
                }
                if model.tiling_config is not None
                else {"enabled": False}
            )

        logger.info("Effective VAE tiling: %s", effective_desc)

    def _save(self, job: Job, image, args, seed: int) -> str:
        from mflux.utils.image_util import ImageUtil

        with Invocation.LOCK:
            Invocation.apply_globals(args)
            path = ImageUtil.resolve_output_path(args.output.format(seed=seed))
            image.save(path=path, export_json_metadata=True, overwrite=True)
        sidecar = {
            "command": job.invocation.command,
            "payload": job.payload,
            "seed": seed,
            "job": job.id,
            "effective_tiling": getattr(job, "tiling_config", {"enabled": bool(getattr(args, "vae_tiling", False))}),
        }
        Path(path).with_suffix(".web.json").write_text(json.dumps(sidecar, indent=2))
        return Path(path).name

    @staticmethod
    def _release_memory() -> None:
        import mlx.core as mx

        gc.collect()
        mx.clear_cache()

    @staticmethod
    def _describe_error(exc: Exception) -> str:
        from mflux.utils.exceptions import MFluxUserException

        if isinstance(exc, (InvocationError, AdapterError, MFluxUserException)):
            return str(exc)
        return f"{type(exc).__name__}: {exc}"
