from argparse import Namespace

import pytest
from mflux.callbacks.callback_manager import CallbackManager
from test_runner import FakeAdapter, FakeInvocation, wait

from mflux.web.seed.invocation import Invocation
from mflux.web.seed.paths import PathGuard
from mflux.web.seed.runner import Job, JobRunner
from mflux.web.seed.schema import FormSchema


@pytest.fixture(scope="module")
def schema():
    return FormSchema()


@pytest.fixture
def guard(tmp_path):
    (tmp_path / "out/.uploads").mkdir(parents=True)
    return PathGuard(tmp_path / "out", tmp_path / "out/.uploads", [], [])


def build(payload, guard, schema, tmp_path):
    return Invocation.from_payload(payload, guard, schema, tmp_path / "out" / "img")


class FakeVAE:
    def __init__(self):
        self.supports_implicit_tiling = True


class FakeZImageModel:
    def __init__(self):
        from mflux.callbacks.callback_registry import CallbackRegistry

        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        self.vae = FakeVAE()


def test_z_image_schema_defaults(schema):
    for cmd in ("mflux-generate-z-image", "mflux-generate-z-image-turbo"):
        fields = {f["flag"]: f for f in schema.command(cmd)["fields"]}
        assert fields["--vae-tiling"]["default"] is True
        assert fields["--vae-tile-size"]["default"] == 512

    z_models = {m["name"]: m["default_steps"] for m in schema.command("mflux-generate-z-image")["models"]}
    assert z_models["z-image"] == 50
    assert z_models["z-image-turbo"] == 8

    turbo_models = {m["name"]: m["default_steps"] for m in schema.command("mflux-generate-z-image-turbo")["models"]}
    assert turbo_models["z-image-turbo"] == 8


def test_z_image_turbo_default_scheduler_and_steps(schema, guard, tmp_path):
    payload = {
        "command": "mflux-generate-z-image",
        "model": {"source": "builtin", "value": "z-image-turbo"},
        "options": {"--prompt": "a futuristic car"},
    }
    inv = build(payload, guard, schema, tmp_path)
    args = inv.parse()
    inv.adapter.validate(args)
    assert args.scheduler == "linear"
    assert args.steps == 8


def test_z_image_base_default_scheduler_and_steps(schema, guard, tmp_path):
    payload = {
        "command": "mflux-generate-z-image",
        "model": {"source": "builtin", "value": "z-image"},
        "options": {"--prompt": "a peaceful lake"},
    }
    inv = build(payload, guard, schema, tmp_path)
    args = inv.parse()
    inv.adapter.validate(args)
    assert args.scheduler == "flow_match_euler_discrete"
    assert args.steps == 50


def test_z_image_preserves_explicit_user_scheduler_and_steps(schema, guard, tmp_path):
    # User explicitly chooses flow_match_euler_discrete and 24 steps on Turbo
    payload_turbo = {
        "command": "mflux-generate-z-image",
        "model": {"source": "builtin", "value": "z-image-turbo"},
        "options": {
            "--prompt": "a futuristic car",
            "--scheduler": "flow_match_euler_discrete",
            "--steps": 24,
        },
    }
    inv_turbo = build(payload_turbo, guard, schema, tmp_path)
    args_turbo = inv_turbo.parse()
    inv_turbo.adapter.validate(args_turbo)
    assert args_turbo.scheduler == "flow_match_euler_discrete"
    assert args_turbo.steps == 24

    # User explicitly chooses linear and 15 steps on Base
    payload_base = {
        "command": "mflux-generate-z-image",
        "model": {"source": "builtin", "value": "z-image"},
        "options": {
            "--prompt": "a peaceful lake",
            "--scheduler": "linear",
            "--steps": 15,
        },
    }
    inv_base = build(payload_base, guard, schema, tmp_path)
    args_base = inv_base.parse()
    inv_base.adapter.validate(args_base)
    assert args_base.scheduler == "linear"
    assert args_base.steps == 15


def test_z_image_tiling_enabled_by_default(tmp_path):
    runner = JobRunner(cache_size=1)
    model = FakeZImageModel()
    inv = FakeInvocation(FakeAdapter(), tmp_path)
    inv.command = "mflux-generate-z-image"
    job = Job(id="test_job", invocation=inv, payload={})
    args = Namespace(web_command="mflux-generate-z-image", web_raw_options={}, vae_tiling=False, vae_tile_size=None)

    runner._apply_tiling(model, args, job=job)
    assert model.tiling_config is not None
    assert model.tiling_config.vae_decode_tile_size == 512
    assert model.tiling_config.vae_decode_overlap == 8
    assert job.tiling_config == {"enabled": True, "tile_size": 512, "overlap": 8}


def test_z_image_tiling_custom_tile_size(tmp_path):
    runner = JobRunner(cache_size=1)
    model = FakeZImageModel()
    inv = FakeInvocation(FakeAdapter(), tmp_path)
    inv.command = "mflux-generate-z-image"
    job = Job(id="test_job", invocation=inv, payload={})
    args = Namespace(web_command="mflux-generate-z-image", web_raw_options={}, vae_tiling=True, vae_tile_size=768)

    runner._apply_tiling(model, args, job=job)
    assert model.tiling_config is not None
    assert model.tiling_config.vae_decode_tile_size == 768
    assert model.tiling_config.vae_decode_overlap == 8
    assert job.tiling_config == {"enabled": True, "tile_size": 768, "overlap": 8}


def test_z_image_tiling_explicitly_disabled(tmp_path):
    runner = JobRunner(cache_size=1)
    model = FakeZImageModel()
    inv = FakeInvocation(FakeAdapter(), tmp_path)
    inv.command = "mflux-generate-z-image"
    job = Job(id="test_job", invocation=inv, payload={})
    args = Namespace(
        web_command="mflux-generate-z-image",
        web_raw_options={"--vae-tiling": False},
        vae_tiling=False,
        vae_tile_size=None,
    )

    runner._apply_tiling(model, args, job=job)
    assert model.tiling_config is None
    assert model.vae.supports_implicit_tiling is False
    assert job.tiling_config == {"enabled": False}


def test_low_ram_does_not_override_explicitly_disabled_tiling(tmp_path):
    runner = JobRunner(cache_size=1)
    model = FakeZImageModel()
    inv = FakeInvocation(FakeAdapter(), tmp_path)
    inv.command = "mflux-generate-z-image"
    job = Job(id="test_job", invocation=inv, payload={})
    args = inv.parse()
    args.web_command = "mflux-generate-z-image"
    args.web_raw_options = {"--vae-tiling": False}
    args.vae_tiling = False
    args.vae_tile_size = None
    args.low_ram = True
    args.mlx_cache_limit_gb = None
    args.prompt = "test"
    args.seed = [42]
    args.verify = False
    args.stepwise_image_output_dir = None

    # 1. Apply tiling initially
    runner._apply_tiling(model, args, job=job)
    assert model.tiling_config is None
    assert model.vae.supports_implicit_tiling is False

    # 2. Register callbacks (as low-ram does in _uncached_model)
    CallbackManager.register_callbacks(args=args, model=model, latent_creator=None)
    assert model.tiling_config is None, (
        "MemorySaver should not implicitly enable tiling when supports_implicit_tiling is False"
    )

    # 3. Post-callback apply tiling guarantee
    runner._apply_tiling(model, args, job=job)
    assert model.tiling_config is None
    assert job.tiling_config == {"enabled": False}


def test_sidecar_records_effective_tiling(tmp_path):
    runner = JobRunner(cache_size=1)
    runner.start()
    try:
        inv = FakeInvocation(FakeAdapter(), tmp_path)
        job = wait(runner.submit(inv, {"command": inv.command}))
        assert job.status == "done"
        sidecar_path = tmp_path / f"{job.images[0].replace('.png', '.web.json')}"
        import json

        sidecar = json.loads(sidecar_path.read_text())
        assert "effective_tiling" in sidecar
        assert sidecar["effective_tiling"]["enabled"] is False
    finally:
        runner.stop()


def test_z_image_tiling_reenabled_on_cached_model_restores_implicit_tiling(tmp_path):
    runner = JobRunner(cache_size=1)
    model = FakeZImageModel()
    assert model.vae.supports_implicit_tiling is True

    inv = FakeInvocation(FakeAdapter(), tmp_path)
    inv.command = "mflux-generate-z-image"

    # 1. First job: explicitly disabled
    job1 = Job(id="job1", invocation=inv, payload={})
    args_disabled = Namespace(
        web_command="mflux-generate-z-image",
        web_raw_options={"--vae-tiling": False},
        vae_tiling=False,
        vae_tile_size=None,
    )
    runner._apply_tiling(model, args_disabled, job=job1)
    assert model.tiling_config is None
    assert model.vae.supports_implicit_tiling is False

    # 2. Second job: re-enabled
    job2 = Job(id="job2", invocation=inv, payload={})
    args_enabled = Namespace(
        web_command="mflux-generate-z-image",
        web_raw_options={"--vae-tiling": True},
        vae_tiling=True,
        vae_tile_size=512,
    )
    runner._apply_tiling(model, args_enabled, job=job2)
    assert model.tiling_config is not None
    assert model.tiling_config.vae_decode_tile_size == 512
    assert model.vae.supports_implicit_tiling is True


def test_explicitly_disabled_tiling_omits_tile_size_from_shell_command(schema, guard, tmp_path):
    payload = {
        "command": "mflux-generate-z-image",
        "options": {
            "--prompt": "cat",
            "--vae-tiling": False,
            "--vae-tile-size": 512,
        },
    }
    inv = build(payload, guard, schema, tmp_path)
    cmd = inv.shell_command()
    assert "--vae-tile-size" not in cmd
    assert "--no-vae-tiling" in cmd or "--vae-tiling" not in cmd
