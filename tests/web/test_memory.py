import time
from argparse import Namespace

import mlx.core as mx
import pytest
from test_runner import FakeAdapter, FakeInvocation, wait

from mflux.web.seed.adapters import ModelCacheKey
from mflux.web.seed.runner import JobRunner


class BufferChurningAdapter(FakeAdapter):
    # Leaves freed MLX buffers behind, like activations do during a real generation.
    def generate(self, model, args, seed, prompt):
        for _ in range(3):
            mx.eval(mx.ones((1024, 1024)) * 2)
        return super().generate(model, args, seed, prompt)


@pytest.fixture
def runner():
    runner = JobRunner(cache_size=1, idle_unload_seconds=0)
    runner.start()
    yield runner
    runner.stop()


def wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.02)
    return predicate()


def test_buffer_cache_is_released_after_every_job(runner, tmp_path):
    job = wait(runner.submit(FakeInvocation(BufferChurningAdapter(), tmp_path), {}))
    assert job.status == "done", job.error
    assert wait_until(lambda: mx.get_cache_memory() == 0)


def test_per_job_cache_limit_does_not_outlive_the_job(runner, tmp_path):
    default = mx.set_cache_limit(0)
    mx.set_cache_limit(default)
    wait(runner.submit(FakeInvocation(FakeAdapter(), tmp_path, mlx_cache_limit_gb=0.5), {}))
    time.sleep(0.05)
    assert mx.set_cache_limit(default) == default


def test_toggling_bake_without_loras_reuses_the_loaded_model(runner, tmp_path):
    adapter = FakeAdapter()
    for bake in (True, False, True):
        wait(runner.submit(FakeInvocation(adapter, tmp_path, bake_lora=bake), {}))
    assert adapter.loads == 1


def test_bake_and_scales_still_matter_with_loras():
    def key(**kwargs):
        base = dict(
            model=None,
            model_path="/m",
            base_model=None,
            quantize=None,
            lora_paths=["a"],
            lora_scales=[1.0],
            bake_lora=True,
        )
        return ModelCacheKey.of("cmd", Namespace(**{**base, **kwargs}))

    assert key() != key(bake_lora=False)
    assert key() != key(lora_scales=[0.5])
    assert key(lora_paths=[], bake_lora=True) == key(lora_paths=[], bake_lora=False, lora_scales=[0.3])


def test_idle_model_is_unloaded(tmp_path):
    runner = JobRunner(cache_size=1, idle_unload_seconds=0.2)
    runner.start()
    try:
        adapter = FakeAdapter()
        wait(runner.submit(FakeInvocation(adapter, tmp_path), {}))
        assert runner.status()["cached_models"]
        assert 0 < runner.status()["unload_in"] <= 0.2
        assert wait_until(lambda: runner.status()["cached_models"] == [])
        assert runner.status()["unload_in"] is None
        wait(runner.submit(FakeInvocation(adapter, tmp_path), {}))
        assert adapter.loads == 2
    finally:
        runner.stop()


def test_unload_request_frees_the_model(runner, tmp_path):
    assert runner.request_unload() is False
    wait(runner.submit(FakeInvocation(FakeAdapter(), tmp_path), {}))
    assert runner.request_unload() is True
    assert wait_until(lambda: runner.status()["cached_models"] == [])


def test_status_reports_memory(runner):
    assert set(runner.status()["memory"]) == {"active_gb", "cache_gb", "peak_gb"}


def test_qwen21_geometry_cache_is_bounded():
    import mlx.nn as nn
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    # The bounded geometry cache (GEOMETRY_CACHE_SIZE + eviction) is a fork
    # enhancement to core; upstream main keeps an unbounded cache. The web UI
    # works with both; this test only asserts the fork's bound.
    if not hasattr(Qwen21Transformer, "GEOMETRY_CACHE_SIZE"):
        import pytest

        pytest.skip("core Qwen21Transformer lacks the fork's bounded geometry cache")

    transformer = Qwen21Transformer.__new__(Qwen21Transformer)
    nn.Module.__init__(transformer)
    transformer._geometry_cache = {}
    transformer.pos_embed = lambda text_len, height, width: (mx.zeros((text_len,)), mx.zeros((text_len,)))
    for text_len in range(1, 50):
        transformer._geometry(text_len=text_len, latent_height=4, latent_width=4, encoder_hidden_states_mask=None)
    assert len(transformer._geometry_cache) == Qwen21Transformer.GEOMETRY_CACHE_SIZE
    assert (49, 4, 4) in transformer._geometry_cache
    assert (1, 4, 4) not in transformer._geometry_cache


def test_jobrunner_baseline_memory_and_cache_limits(tmp_path):
    assert JobRunner.DEFAULT_MEMORY_LIMIT_BYTES == 16 * 1024**3
    assert JobRunner.DEFAULT_CACHE_LIMIT_BYTES == 1 * 1024**3
    runner = JobRunner(cache_size=1, idle_unload_seconds=0)
    runner.start()
    try:
        limits = runner.status()["limits"]
        assert limits["baseline_memory_limit_bytes"] == 17_179_869_184
        assert limits["baseline_cache_limit_bytes"] == 1_073_741_824
        assert limits["memory_limit_gib"] == 16.0
        assert limits["cache_limit_gib"] == 1.0
    finally:
        runner.stop()


def test_jobrunner_limits_restored_after_custom_job_cache_limit(runner, tmp_path):
    job = wait(runner.submit(FakeInvocation(FakeAdapter(), tmp_path, mlx_cache_limit_gb=0.5), {}))
    assert job.status == "done"
    limits = runner.status()["limits"]
    assert limits["cache_limit_bytes"] == 1_073_741_824
    assert limits["memory_limit_bytes"] == 17_179_869_184


def test_jobrunner_limits_restored_after_job_error(runner, tmp_path):
    class FailingAdapter(FakeAdapter):
        def generate(self, model, args, seed, prompt):
            raise RuntimeError("Simulation error during generation")

    job = wait(runner.submit(FakeInvocation(FailingAdapter(), tmp_path, mlx_cache_limit_gb=0.5), {}))
    assert job.status == "error"
    limits = runner.status()["limits"]
    assert limits["cache_limit_bytes"] == 1_073_741_824
    assert limits["memory_limit_bytes"] == 17_179_869_184


def test_jobrunner_limits_restored_after_job_cancellation(runner, tmp_path):
    job = runner.submit(FakeInvocation(FakeAdapter(), tmp_path, mlx_cache_limit_gb=0.5), {})
    runner.cancel(job.id)
    wait(job)
    assert job.status == "cancelled"
    limits = runner.status()["limits"]
    assert limits["cache_limit_bytes"] == 1_073_741_824
    assert limits["memory_limit_bytes"] == 17_179_869_184


def test_excessive_cache_limit_rejected(tmp_path):
    from mflux.web.seed.invocation import Invocation, InvocationError
    from mflux.web.seed.paths import PathGuard
    from mflux.web.seed.schema import FormSchema

    guard = PathGuard(tmp_path / "out", tmp_path / "uploads", [], [])
    schema = FormSchema()
    payload = {
        "command": "mflux-generate-qwen-2.1",
        "options": {
            "--prompt": "x",
            "--mlx-cache-limit-gb": 20.0,
        },
    }
    with pytest.raises(InvocationError, match="must be between 0.1 and 16"):
        Invocation.from_payload(payload, guard, schema, tmp_path / "out/test")


def test_limits_snapshot_does_not_mutate_mlx_limits_and_exposes_gib(runner, monkeypatch):
    limits = runner.status()["limits"]
    assert limits["baseline_memory_limit_gib"] == 16.0
    assert limits["baseline_cache_limit_gib"] == 1.0

    call_count = {"mem": 0, "cache": 0}
    orig_set_mem = mx.set_memory_limit
    orig_set_cache = mx.set_cache_limit

    def tracked_set_mem(val):
        call_count["mem"] += 1
        return orig_set_mem(val)

    def tracked_set_cache(val):
        call_count["cache"] += 1
        return orig_set_cache(val)

    monkeypatch.setattr(mx, "set_memory_limit", tracked_set_mem)
    monkeypatch.setattr(mx, "set_cache_limit", tracked_set_cache)

    for _ in range(5):
        _ = runner.status()
        _ = runner.limits_snapshot()

    assert call_count["mem"] == 0
    assert call_count["cache"] == 0
