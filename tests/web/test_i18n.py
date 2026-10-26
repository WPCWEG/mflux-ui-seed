import re
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.requests import Request

from mflux.web.seed.app import WebApp
from mflux.web.seed.auth import WebAuth
from mflux.web.seed.i18n import (
    COOKIE_NAME,
    DEFAULT_LOCALE,
    SUPPORTED_LOCALES,
    catalogs,
    interpolate,
    jinja_t,
    resolve_locale,
    translate,
)
from mflux.web.seed.settings import WebSettings


def make_request(cookies: dict | None = None, headers: dict | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()],
    }
    req = Request(scope)
    if cookies:
        req._cookies = cookies
    return req


def test_catalogs_loaded_and_valid():
    assert set(catalogs.keys()) == set(SUPPORTED_LOCALES)
    en_keys = set(catalogs["en"].keys())
    zh_keys = set(catalogs["zh-CN"].keys())

    # Keys must match 1:1 across languages
    assert en_keys == zh_keys

    # Core navigation and form keys must be present
    for k in (
        "nav.generate",
        "nav.gallery",
        "language.label",
        "topbar.generate",
        "form.model_family",
        "form.weights",
        "section.memory",
        "section.advanced",
        "form.generate",
        "stage.empty_title",
        "history.title",
    ):
        assert k in en_keys

    # All keys and values must be strings
    for loc in SUPPORTED_LOCALES:
        for k, v in catalogs[loc].items():
            assert isinstance(k, str)
            assert isinstance(v, str)


def test_dynamic_field_and_status_keys():
    en_keys = set(catalogs["en"].keys())
    zh_keys = set(catalogs["zh-CN"].keys())

    required_dynamic_keys = (
        "field.prompt.label",
        "field.negative_prompt.label",
        "field.width.label",
        "field.height.label",
        "field.steps.label",
        "field.guidance.label",
        "field.quantize.label",
        "field.vae_tiling.label",
        "field.vae_tile_size.label",
        "field.mlx_cache_limit_gb.label",
        "field.low_ram.label",
        "field.bake_lora.label",
        "size.more_sizes",
        "size.resolution",
        "size.aspect_ratio_aria",
        "job.status.queued",
        "job.status.loading",
        "job.status.generating",
        "job.status.decoding",
        "job.status.done",
        "job.status.cancelled",
        "job.status.failed",
        "worker.idle",
        "worker.loading",
        "worker.busy",
        "worker.ready",
    )
    for k in required_dynamic_keys:
        assert k in en_keys
        assert k in zh_keys
        # Check that zh-CN translation is not empty and different from key
        assert catalogs["zh-CN"][k]
        assert catalogs["zh-CN"][k] != k


def test_interpolation():
    assert interpolate("Hello {name}!", {"name": "Alice"}) == "Hello Alice!"
    assert interpolate("Step {step}/{total}", {"step": 1, "total": 8}) == "Step 1/8"
    # Missing parameters should preserve placeholder safely
    assert interpolate("Hello {name} and {other}", {"name": "Alice"}) == "Hello Alice and {other}"
    # No parameters
    assert interpolate("Plain text", {}) == "Plain text"


def test_translate_helper():
    assert translate("nav.generate", "en") == "Generate"
    assert translate("nav.generate", "zh-CN") == "生成"
    # Missing key falls back to key itself
    assert translate("nonexistent.key", "zh-CN") == "nonexistent.key"


def test_resolve_locale_cookie_precedence():
    # Cookie wins over Accept-Language
    req = make_request(
        cookies={COOKIE_NAME: "zh-CN"},
        headers={"accept-language": "en-US,en;q=0.9"},
    )
    assert resolve_locale(req) == "zh-CN"

    req_en = make_request(
        cookies={COOKIE_NAME: "en"},
        headers={"accept-language": "zh-CN,zh;q=0.9"},
    )
    assert resolve_locale(req_en) == "en"


def test_resolve_locale_invalid_cookie_falls_through():
    req = make_request(
        cookies={COOKIE_NAME: "fr"},
        headers={"accept-language": "zh-CN,zh;q=0.9"},
    )
    assert resolve_locale(req) == "zh-CN"


def test_resolve_locale_accept_language_q_weights():
    # zh-CN has higher weight
    req1 = make_request(headers={"accept-language": "en;q=0.7,zh-CN;q=0.9"})
    assert resolve_locale(req1) == "zh-CN"

    # en has higher weight
    req2 = make_request(headers={"accept-language": "en-US;q=0.95,zh;q=0.8"})
    assert resolve_locale(req2) == "en"

    # zh-TW normalized to zh-CN
    req3 = make_request(headers={"accept-language": "zh-TW;q=0.9,en;q=0.5"})
    assert resolve_locale(req3) == "zh-CN"


def test_resolve_locale_unsupported_language_falls_to_default():
    req = make_request(headers={"accept-language": "ja-JP,ja;q=0.9,ko;q=0.8"})
    assert resolve_locale(req) == DEFAULT_LOCALE

    # Empty
    req_empty = make_request()
    assert resolve_locale(req_empty) == DEFAULT_LOCALE


def test_jinja_t_context_isolation():
    context_zh = {"locale": "zh-CN"}
    context_en = {"locale": "en"}

    assert jinja_t(context_zh, "nav.generate") == "生成"
    assert jinja_t(context_en, "nav.generate") == "Generate"
    # Unknown key returns key name
    assert jinja_t(context_zh, "unknown.key") == "unknown.key"


def test_templates_i18n_keys_completeness():
    templates_dir = Path(__file__).resolve().parent.parent.parent / "src" / "mflux" / "web" / "seed" / "templates"
    patterns = [
        re.compile(r'data-i18n="([^"]+)"'),
        re.compile(r'data-i18n-placeholder="([^"]+)"'),
        re.compile(r'data-i18n-title="([^"]+)"'),
        re.compile(r'data-i18n-aria="([^"]+)"'),
        re.compile(r"t\('([^']+)'\)"),
    ]

    en_keys = set(catalogs["en"].keys())
    zh_keys = set(catalogs["zh-CN"].keys())

    used_keys = set()
    for html_file in templates_dir.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        for pattern in patterns:
            for match in pattern.finditer(content):
                key = match.group(1)
                # Ignore dynamic Jinja block title_key placeholder
                if not key.startswith("{%"):
                    used_keys.add(key)

    missing_in_en = used_keys - en_keys
    missing_in_zh = used_keys - zh_keys
    assert not missing_in_en, f"Missing keys in en catalog: {missing_in_en}"
    assert not missing_in_zh, f"Missing keys in zh catalog: {missing_in_zh}"


def test_html_serves_locale_and_catalogs(tmp_path):
    settings = WebSettings(
        output_dir=tmp_path / "out",
        config_path=tmp_path / "web.json",
        secret_key="s" * 64,
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    web = WebApp(settings)
    client = TestClient(web.app, base_url="http://127.0.0.1:8000")

    # Default English
    res_en = client.get("/")
    assert res_en.status_code == 200
    assert 'lang="en"' in res_en.text
    assert '<title data-title-key="generate.title">Generate · mflux</title>' in res_en.text
    assert 'data-i18n="nav.generate">Generate<' in res_en.text
    assert 'data-i18n="form.model_family">Model family<' in res_en.text
    assert 'data-i18n="section.memory">Memory<' in res_en.text
    assert 'data-i18n="form.generate">Generate<' in res_en.text
    assert 'id="i18n-data" hidden>' in res_en.text
    assert "private, no-store" in res_en.headers.get("cache-control", "")
    assert "unsafe-inline" not in res_en.headers.get("content-security-policy", "")

    # Cookie zh-CN
    client.cookies.set(COOKIE_NAME, "zh-CN")
    res_zh = client.get("/")
    assert res_zh.status_code == 200
    assert 'lang="zh-CN"' in res_zh.text
    assert '<title data-title-key="generate.title">生成 · mflux</title>' in res_zh.text
    assert 'data-i18n="nav.generate">生成<' in res_zh.text
    assert 'data-i18n="form.model_family">模型架构<' in res_zh.text
    assert 'data-i18n="section.memory">显存与性能<' in res_zh.text
    assert 'data-i18n="form.generate">开始生成<' in res_zh.text
    assert "selected>中文</option>" in res_zh.text


def test_gallery_serves_locale(tmp_path):
    settings = WebSettings(
        output_dir=tmp_path / "out",
        config_path=tmp_path / "web.json",
        secret_key="s" * 64,
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    web = WebApp(settings)
    client = TestClient(web.app, base_url="http://127.0.0.1:8000")

    # English gallery
    res_en = client.get("/gallery")
    assert res_en.status_code == 200
    assert '<title data-title-key="gallery.title">Gallery · mflux</title>' in res_en.text
    assert 'data-i18n="gallery.heading">Gallery<' in res_en.text
    assert 'data-i18n="gallery.reuse">Reuse settings<' in res_en.text

    # Chinese gallery
    client.cookies.set(COOKIE_NAME, "zh-CN")
    res_zh = client.get("/gallery")
    assert res_zh.status_code == 200
    assert '<title data-title-key="gallery.title">画廊 · mflux</title>' in res_zh.text
    assert 'data-i18n="gallery.heading">画廊<' in res_zh.text
    assert 'data-i18n="gallery.reuse">复用参数<' in res_zh.text


def test_auth_pages_serve_locale(tmp_path):
    # Setup page (auth required, no key set)
    settings_setup = WebSettings(
        output_dir=tmp_path / "out_setup",
        config_path=tmp_path / "web_setup.json",
        secret_key="s" * 64,
        require_auth=True,
    )
    settings_setup.output_dir.mkdir(parents=True, exist_ok=True)
    settings_setup.upload_dir.mkdir(parents=True, exist_ok=True)
    web_setup = WebApp(settings_setup)
    client_setup = TestClient(web_setup.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
    client_setup.cookies.set(COOKIE_NAME, "zh-CN")

    res_setup_zh = client_setup.get("/setup")
    assert res_setup_zh.status_code == 200
    assert '<title data-title-key="setup.title">初始设置 · mflux</title>' in res_setup_zh.text
    assert 'data-i18n="setup.heading">创建 API 密钥<' in res_setup_zh.text
    assert 'data-i18n="setup.submit">保存并登录<' in res_setup_zh.text

    # Login page (auth enabled with an api key hash)
    settings_login = WebSettings(
        output_dir=tmp_path / "out_login",
        config_path=tmp_path / "web_login.json",
        secret_key="s" * 64,
        api_key_hash=WebAuth.hash_key("valid-key-at-least-12-chars"),
    )
    settings_login.output_dir.mkdir(parents=True, exist_ok=True)
    settings_login.upload_dir.mkdir(parents=True, exist_ok=True)
    web_login = WebApp(settings_login)
    client_login = TestClient(web_login.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
    client_login.cookies.set(COOKIE_NAME, "zh-CN")

    res_login_zh = client_login.get("/login")
    assert res_login_zh.status_code == 200
    assert '<title data-title-key="login.title">登录 · mflux</title>' in res_login_zh.text
    assert 'data-i18n="login.heading">登录<' in res_login_zh.text
    assert 'data-i18n="login.remember">30 天内保持登录状态<' in res_login_zh.text
    assert 'data-i18n="login.submit">登录<' in res_login_zh.text
