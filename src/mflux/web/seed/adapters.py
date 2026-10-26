import importlib
from argparse import Namespace
from pathlib import Path

from mflux.utils.dimension_resolver import DimensionResolver


class AdapterError(ValueError): ...


class CommandAdapter:
    # Mirrors one CLI main(), split at the point the web runner needs: load() builds the
    # model once (cacheable), generate() runs one seed on it. Keep each subclass in step
    # with its CLI until phase 2 moves this split into the CLIs themselves.
    command: str = ""
    module: str = ""
    latent_creator_path: str = ""
    builtin_models: tuple[str, ...] = ()
    default_guidance: float | None = None
    default_scheduler: str | None = None
    passes_pid: bool = True

    def build_parser(self):
        return importlib.import_module(self.module).build_parser()

    def validate(self, args: Namespace) -> None:
        if args.guidance is None and self.default_guidance is not None:
            args.guidance = self.default_guidance
        if self.default_scheduler is not None and "--scheduler" not in args.web_provided_flags:
            args.scheduler = self.default_scheduler

    def load(self, args: Namespace):
        raise NotImplementedError

    def latent_creator(self):
        module_path, name = self.latent_creator_path.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), name)

    def generate(self, model, args: Namespace, seed: int, prompt: str):
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        kwargs = dict(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=args.guidance,
            scheduler=args.scheduler,
            image_path=args.image_path,
            num_inference_steps=args.steps,
            image_strength=args.image_strength,
            negative_prompt=args.negative_prompt,
        )
        if self.passes_pid:
            kwargs.update(pid_decode=args.pid_decode, pid_degrade_sigma=args.pid_degrade_sigma)
        return model.generate_image(**kwargs)

    @staticmethod
    def lora_kwargs(args: Namespace) -> dict:
        from mflux.cli.parser.parsers import lora_init_kwargs_from_args

        return lora_init_kwargs_from_args(args)

    @staticmethod
    def restricted_config(args: Namespace, registry_key: str, extra_keys: tuple[str, ...] = ()):
        from mflux.models.common.resolution.config_resolution import ConfigResolution

        return ConfigResolution.resolve_restricted(
            args.model, registry_key, model_path=args.model_path, extra_keys=extra_keys, base_model=args.base_model
        )


class FluxAdapter(CommandAdapter):
    command = "mflux-generate"
    module = "mflux.models.flux.cli.flux_generate"
    latent_creator_path = "mflux.models.flux.latent_creator.flux_latent_creator.FluxLatentCreator"
    builtin_models = ("dev", "schnell", "krea-dev")

    def validate(self, args: Namespace) -> None:
        from mflux.cli.defaults import defaults as ui_defaults

        if (args.base_model and "klein" in args.base_model.lower()) or (
            args.model and "flux2-klein" in args.model.lower()
        ):
            raise AdapterError("FLUX.2 Klein is not supported by mflux-generate. Use mflux-generate-flux2 instead.")
        if args.model is None and args.base_model is None:
            raise AdapterError("Choose a model for mflux-generate (dev, schnell, krea-dev or a local checkpoint).")
        if args.guidance is None:
            args.guidance = ui_defaults.GUIDANCE_SCALE

    def load(self, args: Namespace):
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux.variants.txt2img.flux import Flux1

        model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
        return Flux1(
            model_config=model_config, quantize=args.quantize, model_path=args.model_path, **self.lora_kwargs(args)
        )


class Flux2Adapter(CommandAdapter):
    command = "mflux-generate-flux2"
    module = "mflux.models.flux2.cli.flux2_generate"
    latent_creator_path = "mflux.models.flux2.latent_creator.flux2_latent_creator.Flux2LatentCreator"
    default_guidance = 1.0

    @property
    def builtin_models(self) -> tuple[str, ...]:
        cli = importlib.import_module(self.module)
        return (cli.DEFAULT_MODEL, *cli.FAMILY_MODELS)

    def validate(self, args: Namespace) -> None:
        super().validate(args)
        cli = importlib.import_module(self.module)
        model_config = self.restricted_config(args, cli.DEFAULT_MODEL, cli.FAMILY_MODELS)
        if args.guidance != 1.0 and "base" not in model_config.model_name.lower():
            raise AdapterError("Guidance other than 1.0 is only supported for FLUX.2 base models.")
        args.scheduler = "flow_match_euler_discrete"

    def load(self, args: Namespace):
        from mflux.models.flux2.variants import Flux2Klein

        cli = importlib.import_module(self.module)
        model_config = self.restricted_config(args, cli.DEFAULT_MODEL, cli.FAMILY_MODELS)
        return Flux2Klein(
            model_config=model_config, quantize=args.quantize, model_path=args.model_path, **self.lora_kwargs(args)
        )

    def generate(self, model, args: Namespace, seed: int, prompt: str):
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=args.guidance,
            image_path=args.image_path,
            num_inference_steps=args.steps,
            image_strength=args.image_strength,
            scheduler="flow_match_euler_discrete",
            pid_decode=args.pid_decode,
            pid_degrade_sigma=args.pid_degrade_sigma,
        )


class QwenAdapter(CommandAdapter):
    command = "mflux-generate-qwen"
    module = "mflux.models.qwen.cli.qwen_image_generate"
    latent_creator_path = "mflux.models.qwen.latent_creator.qwen_latent_creator.QwenLatentCreator"
    builtin_models = ("qwen-image",)

    def validate(self, args: Namespace) -> None:
        from mflux.cli.defaults import defaults as ui_defaults

        if args.guidance is None:
            args.guidance = ui_defaults.GUIDANCE_SCALE

    def load(self, args: Namespace):
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        return QwenImage(quantize=args.quantize, model_path=args.model_path, **self.lora_kwargs(args))


class Qwen21Adapter(CommandAdapter):
    command = "mflux-generate-qwen-2.1"
    module = "mflux.models.qwen21.cli.qwen21_generate"
    latent_creator_path = "mflux.models.qwen21.latent_creator.qwen21_latent_creator.Qwen21LatentCreator"
    builtin_models = ("qwen-image-2.1",)
    default_guidance = 1.0
    passes_pid = False

    def validate(self, args: Namespace) -> None:
        super().validate(args)
        self.restricted_config(args, "qwen-image-2.1")

    def load(self, args: Namespace):
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        return QwenImage21(
            quantize=args.quantize,
            model_path=args.model_path,
            model_config=self.restricted_config(args, "qwen-image-2.1"),
            **self.lora_kwargs(args),
        )


class ZImageAdapter(CommandAdapter):
    command = "mflux-generate-z-image"
    module = "mflux.models.z_image.cli.z_image_generate"
    latent_creator_path = "mflux.models.z_image.latent_creator.ZImageLatentCreator"
    builtin_models = ("z-image", "z-image-turbo")
    default_scheduler = None  # Resolved dynamically based on the resolved model variant

    def validate(self, args: Namespace) -> None:
        super().validate(args)
        model_config = self.restricted_config(args, "z-image", ("z-image-turbo",))
        is_turbo = not model_config.supports_guidance or "turbo" in model_config.model_name.lower()

        # Scheduler resolution based on the resolved model variant:
        # Z-Image Turbo defaults to linear; Z-Image Base defaults to flow_match_euler_discrete.
        # User explicit choice is preserved.
        if "--scheduler" not in args.web_provided_flags:
            args.scheduler = "linear" if is_turbo else "flow_match_euler_discrete"

        # Steps resolution based on the resolved model variant:
        # Z-Image Turbo defaults to 8; Z-Image Base defaults to 50.
        # User explicit choice is preserved.
        if "--steps" not in args.web_provided_flags or args.steps is None:
            args.steps = 8 if is_turbo else 50

    def load(self, args: Namespace):
        import mlx.core as mx
        from mflux.models.z_image.variants.z_image import ZImage

        model_config = self.restricted_config(args, "z-image", ("z-image-turbo",))
        model = ZImage(
            model_config=model_config,
            quantize=args.quantize,
            model_path=args.model_path,
            float32=getattr(args, "float32", False),
            compute_precision=getattr(args, "compute_precision", None),
            **self.lora_kwargs(args),
        )
        mx.eval(model.parameters())
        return model


class ZImageTurboAdapter(CommandAdapter):
    command = "mflux-generate-z-image-turbo"
    module = "mflux.models.z_image.cli.z_image_turbo_generate"
    latent_creator_path = "mflux.models.z_image.latent_creator.ZImageLatentCreator"
    builtin_models = ("z-image-turbo",)
    default_scheduler = "linear"

    def validate(self, args: Namespace) -> None:
        self._config(args)
        if "--scheduler" not in args.web_provided_flags:
            args.scheduler = "linear"
        if "--steps" not in args.web_provided_flags or args.steps is None:
            args.steps = 8

    def load(self, args: Namespace):
        import mlx.core as mx
        from mflux.models.z_image.variants.z_image import ZImage

        model = ZImage(
            model_config=self._config(args),
            quantize=args.quantize,
            model_path=args.model_path,
            float32=getattr(args, "float32", False),
            compute_precision=getattr(args, "compute_precision", None),
            **self.lora_kwargs(args),
        )
        mx.eval(model.parameters())
        return model

    @staticmethod
    def _config(args: Namespace):
        from mflux.models.common.resolution.config_resolution import ConfigResolution

        # Same call as the CLI: this command does not take --base-model into account.
        return ConfigResolution.resolve_restricted(args.model, "z-image-turbo", model_path=args.model_path)


class Krea2Adapter(CommandAdapter):
    command = "mflux-generate-krea2"
    module = "mflux.models.krea2.cli.krea2_generate"
    latent_creator_path = "mflux.models.krea2.latent_creator.Krea2LatentCreator"
    builtin_models = ("krea-2",)
    default_guidance = 1.0

    def validate(self, args: Namespace) -> None:
        super().validate(args)
        self._config(args)

    def load(self, args: Namespace):
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2

        return Krea2(
            model_config=self._config(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **self.lora_kwargs(args),
        )

    @staticmethod
    def _config(args: Namespace):
        from mflux.models.common.resolution.config_resolution import ConfigResolution

        return ConfigResolution.resolve_restricted(args.model, "krea-2", model_path=args.model_path)


class ErnieImageAdapter(CommandAdapter):
    command = "mflux-generate-ernie-image"
    module = "mflux.models.ernie_image.cli.ernie_image_generate"
    latent_creator_path = "mflux.models.ernie_image.latent_creator.ErnieLatentCreator"
    builtin_models = ("ernie-image",)
    default_guidance = 4.0
    default_scheduler = "linear"
    registry_key = "ernie-image"

    def validate(self, args: Namespace) -> None:
        super().validate(args)
        self._config(args)

    def load(self, args: Namespace):
        from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage

        return ErnieImage(
            model_config=self._config(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **self.lora_kwargs(args),
        )

    def _config(self, args: Namespace):
        from mflux.models.common.resolution.config_resolution import ConfigResolution

        return ConfigResolution.resolve_restricted(args.model, self.registry_key, model_path=args.model_path)


class ErnieImageTurboAdapter(ErnieImageAdapter):
    command = "mflux-generate-ernie-image-turbo"
    module = "mflux.models.ernie_image.cli.ernie_image_turbo_generate"
    builtin_models = ("ernie-image-turbo",)
    default_guidance = 1.0
    default_scheduler = None
    registry_key = "ernie-image-turbo"

    def validate(self, args: Namespace) -> None:
        super().validate(args)
        if args.guidance != 1.0:
            raise AdapterError("Guidance is only supported for base ERNIE-Image. Use 1.0 for turbo.")


ADAPTERS: dict[str, CommandAdapter] = {
    adapter.command: adapter
    for adapter in (
        FluxAdapter(),
        Flux2Adapter(),
        QwenAdapter(),
        Qwen21Adapter(),
        ZImageAdapter(),
        ZImageTurboAdapter(),
        Krea2Adapter(),
        ErnieImageAdapter(),
        ErnieImageTurboAdapter(),
    )
}


class ModelCacheKey:
    @staticmethod
    def of(command: str, args: Namespace) -> tuple:
        lora_paths = tuple(args.lora_paths or ())
        # Scales and baking only change the weights when there are LoRAs; keying on them
        # otherwise reloads an identical model whenever the checkbox is toggled.
        return (
            command,
            args.model,
            args.model_path,
            args.base_model,
            args.quantize,
            lora_paths,
            tuple(args.lora_scales or ()) if lora_paths else (),
            bool(args.bake_lora) if lora_paths else None,
        )

    @staticmethod
    def describe(key: tuple) -> dict:
        command, model, model_path, base_model, quantize, lora_paths, lora_scales, _ = key
        return {
            "command": command,
            "model": model_path and Path(model_path).name or model,
            "base_model": base_model,
            "quantize": quantize,
            "loras": [Path(p).name for p in lora_paths],
            "lora_scales": list(lora_scales),
        }
