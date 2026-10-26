import threading

from mflux.web.seed.adapters import ADAPTERS


class FormSchema:
    # Rendered by dedicated widgets (model picker, LoRA list, seed list, image upload) with
    # their own validation in Invocation; never accepted as a plain option value.
    SPECIAL_FLAGS = ("--model", "--lora", "--image", "--seed")
    # Paths, stdin and output plumbing the server owns. A form must never set these.
    BLOCKED_FLAGS = frozenset(
        {
            "--lora-paths",
            "--lora-scales",
            "--lora-style",
            "--image-path",
            "--image-strength",
            "--output",
            "--stepwise-image-output-dir",
            "--config-from-metadata",
            "--prompt-file",
            "--metadata",
            "--no-metadata",
            "--auto-seeds",
        }
    )
    TEXTAREA_FLAGS = ("--prompt", "--negative-prompt")
    GROUPS = {
        "basic": ("--prompt", "--negative-prompt", "--width", "--height", "--steps", "--guidance"),
        "model": ("--base-model", "--quantize", "--bake-lora"),
        "memory": (
            "--low-ram",
            "--mlx-cache-limit-gb",
            "--vae-tiling",
            "--vae-tile-size",
            "--battery-percentage-stop-limit",
        ),
    }
    LABELS = {
        "--mlx-cache-limit-gb": "MLX cache limit (GiB)",
        "--vae-tiling": "VAE tiling",
        "--vae-tile-size": "VAE tile size",
        "--pid-decode": "PiD decode",
        "--pid-degrade-sigma": "PiD degrade sigma",
        "--low-ram": "Low RAM mode",
        "--bake-lora": "Bake LoRA into weights",
        "--battery-percentage-stop-limit": "Battery stop limit (%)",
    }

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    def commands(self) -> list[dict]:
        return [self.command(name) for name in ADAPTERS]

    def command(self, name: str) -> dict:
        with self._lock:
            if name not in self._cache:
                self._cache[name] = FormSchema._build(name)
            return self._cache[name]

    def fields_by_flag(self, name: str) -> dict[str, dict]:
        return {field["flag"]: field for field in self.command(name)["fields"]}

    @staticmethod
    def _build(name: str) -> dict:
        from mflux.cli.capabilities import describe_command
        from mflux.cli.defaults import defaults as ui_defaults

        adapter = ADAPTERS[name]
        described = describe_command(name, adapter.module)
        if described.get("coverage") != "full":
            return {"command": name, "available": False, "error": described.get("error", "not introspectable")}
        fields = []
        for option in described["options"]:
            field = FormSchema._field(option, adapter)
            if field is not None:
                fields.append(field)
        models = [
            {
                "name": model,
                "default_steps": 8 if "turbo" in model else ui_defaults.model_inference_steps(model),
            }
            for model in adapter.builtin_models
        ]
        return {
            "command": name,
            "available": True,
            "description": described.get("description"),
            "traits": described.get("traits", {}),
            "models": models,
            "fields": fields,
        }

    @staticmethod
    def _field(option: dict, adapter) -> dict | None:
        flag = option["flag"]
        if flag in FormSchema.BLOCKED_FLAGS or option["status"] in ("ignored", "rejected"):
            return None
        field = {
            "flag": flag,
            "label": FormSchema.LABELS.get(flag) or flag.removeprefix("--").replace("-", " ").capitalize(),
            "type": option["type"],
            "default": option.get("parser_default"),
            "help": option.get("help", ""),
            "group": FormSchema._group(flag),
            "status": option["status"],
        }
        if adapter.command.startswith("mflux-generate-z-image"):
            if flag == "--vae-tiling":
                field["default"] = True
            elif flag == "--vae-tile-size":
                field["default"] = 512
        if option["status"] == "conditional":
            field["condition"] = f"Honoured when {option['condition']}. {option.get('reason', '')}".strip()
        if flag in FormSchema.SPECIAL_FLAGS:
            field["widget"] = "special"
            return field
        if option.get("nargs"):
            return None
        if flag == "--base-model":
            return {**field, "widget": "select", "choices": list(adapter.builtin_models)}
        if flag == "--scheduler":
            return {**field, "widget": "select", "choices": FormSchema._schedulers()}
        if option["type"] == "bool":
            negative = f"--no-{flag.removeprefix('--')}"
            if negative in option.get("aliases", []):
                field["negative_flag"] = negative
            return {**field, "widget": "checkbox"}
        if option.get("choices"):
            return {**field, "widget": "select", "choices": option["choices"]}
        if option["type"] in ("int", "float"):
            return {**field, "widget": "number"}
        if option["type"] == "int-or-scale":
            return {**field, "widget": "dimension"}
        if flag in FormSchema.TEXTAREA_FLAGS:
            return {**field, "widget": "textarea"}
        return None

    @staticmethod
    def _group(flag: str) -> str:
        for group, flags in FormSchema.GROUPS.items():
            if flag in flags:
                return group
        return "advanced"

    @staticmethod
    def _schedulers() -> list[str]:
        # Registry names only: the CLI also accepts a dotted import path, which would let a
        # form submission import arbitrary installed modules.
        from mflux.models.common.schedulers import SCHEDULER_REGISTRY

        return [name for name in SCHEDULER_REGISTRY if name.islower() and not name.startswith("seedvr2")]
