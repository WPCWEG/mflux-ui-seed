import argparse
import logging
import os
import sys
from pathlib import Path

from mflux.web.seed.network import NetworkPolicy
from mflux.web.seed.settings import DEFAULT_CONFIG_PATH, DEFAULT_HOST, DEFAULT_OUTPUT_DIR, DEFAULT_PORT, WebSettings

INSTALL_HINT = "mflux-web is missing a dependency; reinstall it: uv tool install --force mflux-web-seed"

LOCAL_YAML_NAME = "mflux-web.yaml"
USER_YAML_PATH = Path.home() / ".config" / "mflux" / LOCAL_YAML_NAME

# CLI (--x) > YAML (x) > built-in defaults. The parser keeps explicit
# defaults below ONLY for --config (a state file, not a setting) and for the
# options with no YAML equivalent; every other option uses a None sentinel so
# "was this set on the command line?" stays answerable.
LOG_LEVELS = ["debug", "info", "warning", "error"]
# argparse dest -> (YAML key, coerce_value flags). merge_settings rejects a YAML key outside this map.
YAML_OPTIONS = {
    "host": ("host", {}),
    "port": ("port", {"is_int": True}),
    "output_dir": ("output_dir", {"is_path": True}),
    "models_dir": ("models_dir", {"is_list": True, "is_path": True}),
    "lora_dir": ("lora_dir", {"is_list": True, "is_path": True}),
    "cache_size": ("cache_size", {"is_int": True}),
    "idle_unload": ("idle_unload", {"is_float": True}),
    "api_key": ("api_key", {}),
    "api_key_file": ("api_key_file", {"is_path": True}),
    "require_auth": ("require_auth", {"is_bool": True}),
    "allowed_host": ("allowed_host", {"is_list": True}),
    "tls_cert": ("tls_cert", {"is_path": True}),
    "tls_key": ("tls_key", {"is_path": True}),
    "behind_https": ("behind_https", {"is_bool": True}),
    "max_upload_mb": ("max_upload_mb", {"is_int": True}),
    "log_level": ("log_level", {}),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a local web UI for mflux image generation.")
    parser.add_argument("--host", default=None, help=f"Bind address (default: {DEFAULT_HOST}). Anything but loopback requires an API key.")  # fmt: off
    parser.add_argument("--port", type=int, default=None, help=f"Port (default: {DEFAULT_PORT}).")
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Where generated images are written; the UI cannot write anywhere else (default: {DEFAULT_OUTPUT_DIR}).")  # fmt: off
    parser.add_argument("--models-dir", type=Path, action="append", default=None, help="Directory whose subdirectories are local model checkpoints. Repeatable.")  # fmt: off
    parser.add_argument("--lora-dir", type=Path, action="append", default=None, help="Directory searched for local .safetensors LoRAs. Repeatable.")  # fmt: off
    parser.add_argument("--cache-size", type=int, default=None, help="How many loaded models to keep in memory between runs (default: 1; 0 reloads every run).")  # fmt: off
    parser.add_argument("--idle-unload", type=float, default=None, metavar="MINUTES", help="Free the loaded model after this many idle minutes (default: 10; 0 keeps it loaded until another model is needed).")  # fmt: off
    parser.add_argument("--api-key", default=None, help="API key for login. Prefer MFLUX_WEB_API_KEY, --api-key-file, or api_key in the YAML file: command lines are visible to other local users.")  # fmt: off
    parser.add_argument("--api-key-file", type=Path, default=None, help="Read the API key from this file.")
    parser.add_argument("--require-auth", action=argparse.BooleanOptionalAction, default=None, help="Require login on loopback too. Without a key, the first visit offers to create one. YAML: require_auth: true/false.")  # fmt: off
    parser.add_argument("--allowed-host", action="append", default=None, help="Extra Host header name to accept (e.g. a Tailscale or reverse-proxy name). Repeatable.")  # fmt: off
    parser.add_argument("--tls-cert", type=Path, default=None, help="Serve HTTPS with this certificate (PEM).")
    parser.add_argument("--tls-key", type=Path, default=None, help="Private key for --tls-cert (PEM).")
    parser.add_argument("--behind-https", action=argparse.BooleanOptionalAction, default=None, help="A TLS-terminating proxy sits in front: mark session cookies Secure. YAML: behind_https: true/false.")  # fmt: off
    parser.add_argument("--max-upload-mb", type=int, default=None, help="Largest init image accepted (default: 50).")
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("MFLUX_WEB_CONFIG", DEFAULT_CONFIG_PATH)), help=f"State file for the session secret and a first-run key (default: {DEFAULT_CONFIG_PATH}). Not read from YAML.")  # fmt: off
    parser.add_argument("--yaml", type=Path, default=None, help=f"YAML settings file. Without it: $MFLUX_WEB_YAML, else ./{LOCAL_YAML_NAME}, else {USER_YAML_PATH}, if it exists. Command-line options override YAML values.")  # fmt: off
    parser.add_argument(
        "--log-level", default=None, choices=LOG_LEVELS, help="debug | info | warning | error (default: info)."
    )
    return parser


def find_yaml(cli_path: Path | None, environ, cwd: Path) -> Path | None:
    """Pick the YAML file: --yaml, then MFLUX_WEB_YAML, then ./mflux-web.yaml, then ~/.config/mflux/mflux-web.yaml.

    A file named by --yaml or MFLUX_WEB_YAML must exist. The two default locations are optional.
    """
    explicit = cli_path or (Path(environ["MFLUX_WEB_YAML"]) if environ.get("MFLUX_WEB_YAML") else None)
    if explicit is not None:
        explicit = explicit.expanduser()
        if not explicit.is_file():
            raise SystemExit(f"mflux-web: YAML settings file not found: {explicit}")
        return explicit
    return next((path for path in (cwd / LOCAL_YAML_NAME, USER_YAML_PATH) if path.is_file()), None)


def load_yaml(yaml_path: Path | None) -> dict:
    """Return parsed settings from the YAML file, or {} if absent/empty.

    mflux-web ships no yaml parser: the config only needs plain scalars and
    lists, so we parse the minimal subset (key: value, lists via hyphen
    bullets) rather than take a PyYAML dependency.
    """
    if yaml_path is None or not yaml_path.is_file():
        return {}
    values: dict = {}
    current_list_key: str | None = None
    for raw_line in yaml_path.read_text().splitlines():
        line = strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- "):
            if current_list_key is None:
                raise SystemExit(f"mflux-web: {yaml_path}: list item outside a list: {raw_line!r}")
            values.setdefault(current_list_key, []).append(line.lstrip()[2:].strip().strip("'\""))
            continue
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise SystemExit(f"mflux-web: {yaml_path}: expected 'key: value', got: {raw_line!r}")
        key, value = key.strip(), value.strip().strip("'\"")
        if not value:
            # "key:" with nothing after it starts a bullet list.
            current_list_key = key
            values.setdefault(current_list_key, [])
        else:
            current_list_key = None
            values[key] = value
    return values


def strip_comment(line: str) -> str:
    # Like YAML, "#" starts a comment only at the start of the line or after whitespace,
    # and never inside quotes: API keys and paths may contain "#".
    quote = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def coerce_value(
    value,
    *,
    is_path: bool = False,
    is_int: bool = False,
    is_float: bool = False,
    is_bool: bool = False,
    is_list: bool = False,
):
    """Coerce a raw YAML string (or list of them) to the type argparse would have produced."""
    if is_list:
        items = value if isinstance(value, list) else [value]
        coerced = [str(item) for item in items]
        return [Path(item).expanduser() for item in coerced] if is_path else coerced
    text = " ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    if is_bool:
        return text.strip().lower() in ("true", "yes", "on", "1")
    if is_path:
        return Path(text).expanduser()
    if is_float:
        return float(text)
    if is_int:
        return int(text)
    return text


def merge_settings(args: argparse.Namespace, yaml_values: dict) -> argparse.Namespace:
    """CLI wins over YAML; YAML wins over built-in defaults."""
    known = {yaml_key for yaml_key, _ in YAML_OPTIONS.values()}
    if unknown := sorted(set(yaml_values) - known):
        raise SystemExit(f"mflux-web: unknown YAML setting(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
    for dest, (yaml_key, flags) in YAML_OPTIONS.items():
        cli_value = getattr(args, dest)
        if cli_value is not None:
            continue
        yaml_raw = yaml_values.get(yaml_key)
        if yaml_raw is None or yaml_raw == []:
            continue
        setattr(args, dest, coerce_value(yaml_raw, **flags))
    # Fill remaining Nones with built-in defaults (outside YAML's reach).
    if args.host is None:
        args.host = DEFAULT_HOST
    if args.port is None:
        args.port = DEFAULT_PORT
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR
    if args.models_dir is None:
        args.models_dir = []
    if args.lora_dir is None:
        args.lora_dir = []
    if args.cache_size is None:
        args.cache_size = 1
    if args.idle_unload is None:
        args.idle_unload = 10
    if args.require_auth is None:
        args.require_auth = False
    if args.allowed_host is None:
        args.allowed_host = []
    if args.behind_https is None:
        args.behind_https = False
    if args.max_upload_mb is None:
        args.max_upload_mb = 50
    if args.log_level is None:
        args.log_level = "info"
    args.log_level = str(args.log_level).lower()
    if args.log_level not in LOG_LEVELS:
        raise SystemExit(f"mflux-web: log_level must be one of {', '.join(LOG_LEVELS)}, got {args.log_level!r}")
    return args


def resolve_api_key(cli: argparse.Namespace, merged: argparse.Namespace, environ) -> str | None:
    """Pick the API key: command line > MFLUX_WEB_API_KEY > YAML. A key file counts at its own level.

    `cli` is the namespace before merge_settings, so the YAML values are still None in it.
    """
    for key, key_file in (
        (cli.api_key, cli.api_key_file),
        (environ.get("MFLUX_WEB_API_KEY"), None),
        (merged.api_key, merged.api_key_file),
    ):
        if key and key_file:
            raise SystemExit("mflux-web: give an API key or an API key file, not both")
        if key_file:
            try:
                return key_file.expanduser().read_text().strip()
            except OSError as exc:
                raise SystemExit(f"mflux-web: cannot read the API key file {key_file}: {exc.strerror}") from None
        if key:
            return key
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        import uvicorn

        from mflux.web.seed.app import WebApp
        from mflux.web.seed.auth import WebAuth
    except ImportError as exc:
        parser.exit(1, f"{INSTALL_HINT}\n({exc})\n")

    cli_args = argparse.Namespace(**vars(args))
    yaml_path = find_yaml(args.yaml, os.environ, Path.cwd())
    yaml_values = load_yaml(yaml_path)
    args = merge_settings(args, yaml_values)

    if (args.tls_cert is None) != (args.tls_key is None):
        parser.error("--tls-cert and --tls-key must be given together")

    api_key = resolve_api_key(cli_args, args, os.environ)
    if api_key and (problem := WebAuth.validate_new_key(api_key)):
        parser.error(problem)

    settings = WebSettings(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir.expanduser().resolve(),
        models_dirs=[d.expanduser().resolve() for d in args.models_dir],
        lora_dirs=[d.expanduser().resolve() for d in args.lora_dir],
        cache_size=args.cache_size,
        idle_unload_minutes=max(args.idle_unload, 0),
        require_auth=args.require_auth,
        api_key_hash=WebAuth.hash_key(api_key) if api_key else None,
        allowed_hosts=args.allowed_host,
        tls_certfile=args.tls_cert,
        tls_keyfile=args.tls_key,
        behind_https=args.behind_https,
        max_upload_mb=args.max_upload_mb,
        config_path=args.config.expanduser(),
    )
    settings.load_persisted_api_key()
    settings.ensure_secret_key()

    if problem := NetworkPolicy.startup_error(
        settings.host, settings.auth_configured, settings.allowed_hosts, settings.behind_https
    ):
        parser.exit(2, f"mflux-web: {problem}\n")

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(WebCli.banner(settings, yaml_path), file=sys.stderr)
    config = uvicorn.Config(
        WebApp(settings).app,
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        ssl_certfile=str(settings.tls_certfile) if settings.tls_certfile else None,
        ssl_keyfile=str(settings.tls_keyfile) if settings.tls_keyfile else None,
        proxy_headers=False,
        timeout_graceful_shutdown=15,
    )
    # uvicorn.Config sets up its loggers on construction, so the filter goes on afterwards.
    if args.log_level != "debug":
        logging.getLogger("uvicorn.access").addFilter(QuietAccessLog())
    try:
        uvicorn.Server(config).run()
    except KeyboardInterrupt:
        # uvicorn shuts down cleanly on Ctrl+C, then re-raises the signal; uvicorn.run()
        # swallows that final KeyboardInterrupt the same way.
        pass


class QuietAccessLog(logging.Filter):
    # The page polls status and loads thumbnails constantly; only --log-level debug shows them.
    NOISY_PREFIXES = (
        "/api/status",
        "/api/session",
        "/api/jobs",
        "/api/images/",
        "/api/gallery",
        "/static/",
        "/favicon.svg",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        method, path = str(args[1]), str(args[2])
        if method != "GET":
            return True
        return not path.startswith(QuietAccessLog.NOISY_PREFIXES)


class WebCli:
    @staticmethod
    def banner(settings: WebSettings, yaml_path: Path | None = None) -> str:
        scheme = "https" if settings.tls_certfile else "http"
        shown_host = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
        lines = [
            f"mflux-web: {scheme}://{shown_host}:{settings.port}",
            f"  settings: {yaml_path or 'none (no YAML file found)'}",
            f"  outputs: {settings.output_dir}",
        ]
        unload = (
            f"after {settings.idle_unload_minutes:g} idle min"
            if settings.idle_unload_minutes
            else "only when another model is needed"
        )
        lines.append(f"  models:  keep {settings.cache_size} loaded, unload {unload}")
        lines.extend(f"  from:    {directory}" for directory in settings.models_dirs)
        lines.extend(f"  loras:   {directory}" for directory in settings.lora_dirs)
        if settings.auth_configured or settings.require_auth:
            lines.append("  auth:    login required")
        else:
            lines.append("  auth:    none (loopback only)")
        if not NetworkPolicy.is_loopback_host(settings.host) and not (settings.tls_certfile or settings.behind_https):
            lines.append(
                "  WARNING: serving plain HTTP beyond loopback; the API key and images cross the network unencrypted. "
                "Use --tls-cert/--tls-key, or a TLS proxy such as `tailscale serve` with --behind-https."
            )
        return "\n".join(lines)


if __name__ == "__main__":
    main()
