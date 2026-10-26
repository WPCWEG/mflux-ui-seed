import json
import re
from pathlib import Path
from typing import Any

from jinja2 import pass_context
from starlette.requests import Request

SUPPORTED_LOCALES = ("en", "zh-CN")
DEFAULT_LOCALE = "en"
COOKIE_NAME = "mflux_ui_lang"
LOCALES_PATH = Path(__file__).parent / "locales.json"

TOKEN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_catalogs() -> dict[str, dict[str, str]]:
    if not LOCALES_PATH.is_file():
        raise RuntimeError(f"Locales file not found: {LOCALES_PATH}")
    try:
        raw = json.loads(LOCALES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid locales JSON in {LOCALES_PATH}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"Locales catalog must be a JSON object, got {type(raw).__name__}")

    catalogs: dict[str, dict[str, str]] = {}
    for loc in SUPPORTED_LOCALES:
        if loc not in raw or not isinstance(raw[loc], dict):
            raise RuntimeError(f"Locales catalog missing required branch for '{loc}'")
        branch: dict[str, str] = {}
        for k, v in raw[loc].items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise RuntimeError(f"Locale '{loc}' item '{k}' has non-string key or value")
            branch[k] = v
        catalogs[loc] = branch

    return catalogs


catalogs = _load_catalogs()


def resolve_locale(request: Request) -> str:
    # 1. Cookie priority: valid mflux_ui_lang
    cookie_val = request.cookies.get(COOKIE_NAME)
    if cookie_val in SUPPORTED_LOCALES:
        return cookie_val

    # 2. Accept-Language header parsing with q weights
    accept = request.headers.get("accept-language", "")
    if accept:
        matched = _parse_accept_language(accept)
        if matched is not None:
            return matched

    # 3. Default fallback
    return DEFAULT_LOCALE


def _parse_accept_language(header: str) -> str | None:
    # Format example: zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7
    parsed: list[tuple[float, str]] = []
    for item in header.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(";")
        tag = parts[0].strip().lower()
        q = 1.0
        for param in parts[1:]:
            param = param.strip()
            if param.startswith("q="):
                try:
                    q = float(param[2:])
                except ValueError:
                    q = 0.0
        if q > 0:
            parsed.append((q, tag))

    # Highest q first
    parsed.sort(key=lambda x: x[0], reverse=True)

    for _, tag in parsed:
        if tag.startswith("zh"):
            return "zh-CN"
        if tag.startswith("en"):
            return "en"

    return None


def interpolate(text: str, params: dict[str, Any]) -> str:
    return TOKEN.sub(lambda m: str(params[m[1]]) if m[1] in params else m[0], text)


def template_context(request: Request) -> dict[str, Any]:
    return {
        "locale": resolve_locale(request),
        "i18n_catalogs": catalogs,
    }


@pass_context
def jinja_t(context: dict[str, Any], key: str, **params: Any) -> str:
    locale = context.get("locale", DEFAULT_LOCALE)
    cat = catalogs.get(locale, catalogs.get(DEFAULT_LOCALE, {}))
    fallback = catalogs.get(DEFAULT_LOCALE, {})
    text = cat.get(key, fallback.get(key, key))
    return interpolate(text, params)


def translate(key: str, locale: str = DEFAULT_LOCALE, **params: Any) -> str:
    cat = catalogs.get(locale, catalogs.get(DEFAULT_LOCALE, {}))
    fallback = catalogs.get(DEFAULT_LOCALE, {})
    text = cat.get(key, fallback.get(key, key))
    return interpolate(text, params)
