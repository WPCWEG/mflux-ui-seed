import contextlib
import io
import math
import re
import shlex
import sys
import threading
import warnings
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

from mflux.web.seed.adapters import ADAPTERS, AdapterError, CommandAdapter
from mflux.web.seed.paths import PathGuard, PathRejected
from mflux.web.seed.schema import FormSchema


class InvocationError(ValueError): ...


@dataclass
class Invocation:
    command: str
    argv: list[str]
    output_stem: Path
    lora_argv: list[str] = field(default_factory=list)
    provided_flags: set[str] = field(default_factory=set)
    raw_options: dict = field(default_factory=dict)

    # parse_args() reads sys.argv and flips two process-wide flags, so every parse and every
    # save that depends on those flags runs under this lock.
    LOCK = threading.RLock()
    SCALE_PATTERN = re.compile(r"^(auto|\d{1,5}|\d{1,2}(\.\d{1,3})?x)$")
    # Upper bounds keep one request from exhausting memory or the queue; far above what any
    # supported model is used with.
    NUMBER_BOUNDS = {
        "--steps": (1, 1000),
        "--guidance": (-100, 100),
        "--mlx-cache-limit-gb": (0.1, 16.0),
        "--vae-tile-size": (128, 8192),
        "--battery-percentage-stop-limit": (1, 99),
        "--pid-degrade-sigma": (0, 100),
    }
    MAX_DIMENSION = 8192
    MAX_SCALE = 8.0
    MAX_SEED = 2**63 - 1

    @staticmethod
    def from_payload(payload: dict, guard: PathGuard, schema: FormSchema, output_stem: Path) -> "Invocation":
        command = payload.get("command")
        adapter = ADAPTERS.get(command) if isinstance(command, str) else None
        if adapter is None:
            raise InvocationError(f"Command not available in the web UI: {command!r}")
        fields = schema.fields_by_flag(command)
        argv: list[str] = []
        provided: set[str] = set()

        options = payload.get("options") or {}
        if not isinstance(options, dict):
            raise InvocationError("options must be an object")
        tiling_disabled = options.get("--vae-tiling") is False
        for flag, value in options.items():
            spec = fields.get(flag)
            if spec is None:
                raise InvocationError(f"Option not allowed: {flag}")
            if flag == "--vae-tile-size" and tiling_disabled:
                continue
            tokens = Invocation._option_tokens(spec, value)
            if tokens:
                argv.extend(tokens)
                provided.add(flag)

        argv.extend(Invocation._model_tokens(payload.get("model"), adapter, guard, provided))
        argv.extend(Invocation._seed_tokens(payload.get("seeds"), provided))
        argv.extend(Invocation._image_tokens(payload.get("image"), guard, fields, provided))
        lora_argv = Invocation._lora_tokens(payload.get("loras"), guard, fields, provided)

        prompt = options.get("--prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvocationError("A prompt is required")
        if prompt.strip() == "-":
            raise InvocationError("A prompt of '-' (read from stdin) is not supported in the web UI")

        argv.extend(["--metadata", f"--output={output_stem}.png"])
        return Invocation(command, argv, output_stem, lora_argv, provided, raw_options=dict(options))

    @property
    def adapter(self) -> CommandAdapter:
        return ADAPTERS[self.command]

    def parse(self, include_loras: bool = True) -> Namespace:
        argv = self.argv + (self.lora_argv if include_loras else [])
        with Invocation.LOCK, Invocation._restored_globals():
            args, messages = Invocation._parse(self.adapter, self.command, argv)
        args.web_command = self.command
        args.web_provided_flags = set(self.provided_flags)
        args.web_raw_options = dict(self.raw_options)
        args.web_messages = messages
        return args

    def validate(self) -> list[str]:
        # LoRA arguments are left out: resolving them may download from the Hub, which the
        # worker does when the job actually runs. PathGuard has already vetted them.
        args = self.parse(include_loras=False)
        if getattr(args, "mlx_cache_limit_gb", None) is not None:
            if not (0.1 <= args.mlx_cache_limit_gb <= 16.0):
                raise InvocationError(
                    f"--mlx-cache-limit-gb ({args.mlx_cache_limit_gb:g} GiB) exceeds the host memory guideline of 16 GiB"
                )
        try:
            self.adapter.validate(args)
        except (AdapterError, ValueError) as exc:
            raise InvocationError(str(exc)) from exc
        return args.web_messages

    def shell_command(self) -> str:
        return shlex.join([self.command, *self.argv, *self.lora_argv])

    @staticmethod
    def apply_globals(args: Namespace) -> None:
        from mflux.utils.generated_image import GeneratedImage
        from mflux.utils.image_util import ImageUtil

        GeneratedImage.model_path = args.model_path
        ImageUtil.embed_metadata_enabled = not getattr(args, "no_metadata", False)

    @staticmethod
    def _parse(adapter: CommandAdapter, command: str, argv: list[str]) -> tuple[Namespace, list[str]]:
        stderr = io.StringIO()
        saved_argv = sys.argv
        sys.argv = [command, *argv]
        try:
            with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stderr(stderr):
                warnings.simplefilter("always")
                args = adapter.build_parser().parse_args()
        except SystemExit as exc:
            raise InvocationError(Invocation._argparse_message(stderr.getvalue(), exc.code)) from None
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user as a validation error
            raise InvocationError(str(exc)) from exc
        finally:
            sys.argv = saved_argv
        return args, [str(w.message) for w in caught]

    @staticmethod
    def _argparse_message(stderr: str, code) -> str:
        # argparse prints usage first, then "prog: error: ..." possibly followed by Tip lines.
        marker = ": error: "
        position = stderr.find(marker)
        if position == -1:
            return stderr.strip() or f"Invalid arguments (exit {code})"
        return stderr[position + len(marker) :].strip()

    @staticmethod
    @contextlib.contextmanager
    def _restored_globals():
        from mflux.utils.generated_image import GeneratedImage
        from mflux.utils.image_util import ImageUtil

        saved = (GeneratedImage.model_path, ImageUtil.embed_metadata_enabled)
        try:
            yield
        finally:
            GeneratedImage.model_path, ImageUtil.embed_metadata_enabled = saved

    @staticmethod
    def _option_tokens(spec: dict, value) -> list[str]:
        flag = spec["flag"]
        kind = spec["widget"]
        if value is None or value == "":
            return []
        if kind == "checkbox":
            if not isinstance(value, bool):
                raise InvocationError(f"{flag} must be true or false")
            if spec.get("negative_flag"):
                return [] if value == spec.get("default") else [flag if value else spec["negative_flag"]]
            return [flag] if value else []
        if kind == "select":
            if str(value) not in spec["choices"]:
                raise InvocationError(f"{flag} must be one of {', '.join(spec['choices'])}")
            return [f"{flag}={value}"]
        if kind == "number":
            number = Invocation._finite_number(flag, value)
            if spec["type"] == "int" and not number.is_integer():
                raise InvocationError(f"{flag} must be a whole number")
            low, high = Invocation.NUMBER_BOUNDS.get(flag, (-1e6, 1e6))
            if not low <= number <= high:
                raise InvocationError(f"{flag} must be between {low} and {high}")
            return [f"{flag}={int(number) if spec['type'] == 'int' else number}"]
        if kind == "dimension":
            text = str(value).strip().lower()
            if not Invocation.SCALE_PATTERN.match(text):
                raise InvocationError(f"{flag} must be a number of pixels, 'auto', or a scale like '2x'")
            if text.endswith("x") and not 0 < float(text[:-1]) <= Invocation.MAX_SCALE:
                raise InvocationError(f"{flag} scale must be above 0x and at most {Invocation.MAX_SCALE:g}x")
            if text.isdigit() and not 16 <= int(text) <= Invocation.MAX_DIMENSION:
                raise InvocationError(f"{flag} must be between 16 and {Invocation.MAX_DIMENSION} pixels")
            return [f"{flag}={text}"]
        if kind in ("text", "textarea"):
            if not isinstance(value, str):
                raise InvocationError(f"{flag} must be text")
            if len(value) > 20_000:
                raise InvocationError(f"{flag} is too long")
            if "\x00" in value:
                raise InvocationError(f"{flag} must not contain NUL characters")
            return [f"{flag}={value}"]
        raise InvocationError(f"Unsupported option: {flag}")

    @staticmethod
    def _finite_number(flag: str, value) -> float:
        # json.loads accepts NaN and Infinity, which would sail through every range check.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvocationError(f"{flag} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise InvocationError(f"{flag} must be a finite number")
        return number

    @staticmethod
    def _model_tokens(model: dict | None, adapter: CommandAdapter, guard: PathGuard, provided: set[str]) -> list[str]:
        if not model:
            return []
        if not isinstance(model, dict):
            raise InvocationError("model must be an object")
        if not model.get("value"):
            return []
        source, value = model.get("source"), str(model["value"])
        try:
            if source == "builtin":
                if value not in adapter.builtin_models:
                    raise InvocationError(f"{value!r} is not a model {adapter.command} runs")
                resolved = value
            elif source == "local":
                resolved = str(guard.local_model(value))
            elif source == "hf":
                resolved = guard.hf_repo(value)
            else:
                raise InvocationError(f"Unknown model source: {source!r}")
        except PathRejected as exc:
            raise InvocationError(str(exc)) from exc
        provided.add("--model")
        return [f"--model={resolved}"]

    @staticmethod
    def _seed_tokens(seeds, provided: set[str]) -> list[str]:
        if seeds in (None, "", []):
            return []
        if isinstance(seeds, (int, str)):
            seeds = [seeds]
        if not isinstance(seeds, list) or len(seeds) > 64:
            raise InvocationError("seeds must be a list of up to 64 integers")
        parsed = []
        for seed in seeds:
            text = str(seed).strip()
            if not text.isdigit() or not text.isascii() or int(text) > Invocation.MAX_SEED:
                raise InvocationError(f"Invalid seed: {seed!r}")
            parsed.append(text)
        provided.add("--seed")
        return ["--seed", *parsed]

    @staticmethod
    def _image_tokens(image: dict | None, guard: PathGuard, fields: dict, provided: set[str]) -> list[str]:
        if not image:
            return []
        if not isinstance(image, dict):
            raise InvocationError("image must be an object")
        if not image.get("upload"):
            return []
        if "--image" not in fields:
            raise InvocationError("This command does not take an init image")
        try:
            path = guard.upload_file(str(image["upload"]))
        except PathRejected as exc:
            raise InvocationError(str(exc)) from exc
        tokens = ["--image", str(path)]
        strength = image.get("strength")
        if strength is not None:
            if not 0 <= Invocation._finite_number("Image strength", strength) <= 1:
                raise InvocationError("Image strength must be between 0 and 1")
            tokens.append(str(float(strength)))
        provided.add("--image")
        return tokens

    @staticmethod
    def _lora_tokens(loras, guard: PathGuard, fields: dict, provided: set[str]) -> list[str]:
        if not loras:
            return []
        if "--lora" not in fields:
            raise InvocationError("This command does not take LoRAs")
        if not isinstance(loras, list) or len(loras) > 8:
            raise InvocationError("loras must be a list of up to 8 entries")
        tokens: list[str] = []
        for entry in loras:
            if not isinstance(entry, dict) or not entry.get("path"):
                raise InvocationError("Each LoRA needs a path")
            try:
                path = guard.lora(str(entry["path"]))
            except PathRejected as exc:
                raise InvocationError(str(exc)) from exc
            scale = Invocation._finite_number("LoRA scale", entry.get("scale", 1.0))
            if not -4 <= scale <= 4:
                raise InvocationError("LoRA scale must be between -4 and 4")
            tokens.extend(["--lora", path, str(scale)])
        provided.add("--lora")
        return tokens
