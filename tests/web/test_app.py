import io

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from mflux.web.seed.app import WebApp  # noqa: E402
from mflux.web.seed.auth import SESSION_COOKIE_NAME, WebAuth  # noqa: E402
from mflux.web.seed.settings import WebSettings  # noqa: E402

KEY = "correct-horse-battery"
BASE = "http://127.0.0.1:8001"


def make_app(tmp_path, **overrides):
    settings = WebSettings(
        output_dir=tmp_path / "out", config_path=tmp_path / "web.json", secret_key="s" * 64, **overrides
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return WebApp(settings)


def client_for(web, peer="127.0.0.1"):
    return TestClient(web.app, base_url=BASE, client=(peer, 50000))


def csrf(client):
    return {"X-MFlux-CSRF": client.get("/api/session").json()["csrf"]}


def png_bytes(color=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_open_loopback_serves_pages_with_security_headers(tmp_path):
    client = client_for(make_app(tmp_path))
    response = client.get("/")
    assert response.status_code == 200
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-frame-options"] == "DENY"
    assert client.get("/static/app.css").status_code == 200


def test_foreign_host_header_is_refused(tmp_path):
    client = TestClient(make_app(tmp_path).app, base_url="http://evil.example")
    assert client.get("/").status_code == 421


def test_unsafe_requests_need_csrf_token(tmp_path):
    client = client_for(make_app(tmp_path))
    body = {"command": "mflux-generate-qwen-2.1", "options": {"--prompt": "x"}}
    assert client.post("/api/validate", json=body).status_code == 403
    assert client.post("/api/validate", json=body, headers={"X-MFlux-CSRF": "forged"}).status_code == 403
    assert client.post("/api/validate", json=body, headers=csrf(client)).status_code == 200


def test_cross_origin_post_is_refused_even_with_token(tmp_path):
    client = client_for(make_app(tmp_path))
    headers = {**csrf(client), "Origin": "http://evil.example"}
    assert client.post("/api/validate", json={}, headers=headers).status_code == 403


def test_auth_redirects_pages_and_rejects_api(tmp_path):
    client = client_for(make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY)))
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"
    assert client.get("/api/commands").status_code == 401
    assert client.get("/login").status_code == 200


def test_login_with_session_cookie_then_logout(tmp_path):
    client = client_for(make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY)))
    assert client.post("/api/login", json={"api_key": "wrong-key-value"}, headers=csrf(client)).status_code == 401
    response = client.post("/api/login", json={"api_key": KEY}, headers=csrf(client))
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert client.get("/api/commands").status_code == 200
    client.post("/api/logout", headers=csrf(client))
    assert client.get("/api/commands").status_code == 401


def test_session_cookie_marked_secure_behind_https(tmp_path):
    client = client_for(make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY), behind_https=True))
    response = client.post("/api/login", json={"api_key": KEY}, headers=csrf(client))
    assert "Secure" in response.headers["set-cookie"]


def test_login_is_throttled(tmp_path):
    client = client_for(make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY)))
    for _ in range(6):
        client.post("/api/login", json={"api_key": "wrong-key-value"}, headers=csrf(client))
    response = client.post("/api/login", json={"api_key": KEY}, headers=csrf(client))
    assert response.status_code == 429


def test_bearer_key_works_without_cookie_or_csrf(tmp_path):
    client = client_for(make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY)))
    headers = {"Authorization": f"Bearer {KEY}"}
    assert client.get("/api/status", headers=headers).status_code == 200
    body = {"command": "mflux-generate-qwen-2.1", "options": {"--prompt": "x"}}
    assert client.post("/api/validate", json=body, headers=headers).status_code == 200
    assert client.get("/api/status", headers={"Authorization": "Bearer wrong-key-value"}).status_code == 401


def test_first_run_setup_only_over_loopback(tmp_path):
    web = make_app(tmp_path, require_auth=True)
    remote = client_for(web, peer="192.168.1.20")
    body = {"api_key": KEY, "api_key_confirm": KEY}
    assert remote.post("/api/setup", json=body, headers=csrf(remote)).status_code == 403
    local = client_for(web)
    assert local.get("/", follow_redirects=False).headers["location"] == "/setup"
    assert (
        local.post(
            "/api/setup", json={"api_key": KEY, "api_key_confirm": "different-key-1"}, headers=csrf(local)
        ).status_code
        == 400
    )
    assert local.post("/api/setup", json=body, headers=csrf(local)).status_code == 200
    assert WebSettings.load_persisted(tmp_path / "web.json")["api_key_hash"].startswith("pbkdf2_sha256$")
    assert local.get("/api/commands").status_code == 200
    assert local.post("/api/setup", json=body, headers=csrf(local)).status_code == 403


def test_upload_is_reencoded_and_rejects_non_images(tmp_path):
    web = make_app(tmp_path)
    client = client_for(web)
    ok = client.post("/api/uploads", files={"file": ("a.png", png_bytes(), "image/png")}, headers=csrf(client))
    assert ok.status_code == 200
    stored = web.settings.upload_dir / ok.json()["id"]
    assert Image.open(stored).size == (16, 16)
    bad = client.post("/api/uploads", files={"file": ("a.png", b"<script>", "image/png")}, headers=csrf(client))
    assert bad.status_code == 400


def test_generate_queues_job_and_exposes_it(tmp_path):
    web = make_app(tmp_path)
    client = client_for(web)
    body = {"command": "mflux-generate-qwen-2.1", "options": {"--prompt": "teapot"}, "seeds": ["5"]}
    job = client.post("/api/generate", json=body, headers=csrf(client)).json()
    assert job["status"] == "queued" and "--seed 5" in job["shell_command"]
    assert client.get(f"/api/jobs/{job['id']}").json()["prompt"] == "teapot"
    assert client.get("/api/jobs").json()["jobs"][0]["id"] == job["id"]
    assert client.post(f"/api/jobs/{job['id']}/cancel", headers=csrf(client)).json() == {"cancelled": True}
    assert client.get("/api/jobs/missing").status_code == 404


def test_generate_rejects_invalid_payload_with_400(tmp_path):
    client = client_for(make_app(tmp_path))
    body = {"command": "mflux-generate-qwen-2.1", "options": {"--prompt": "x", "--output": "/etc/passwd"}}
    response = client.post("/api/generate", json=body, headers=csrf(client))
    assert response.status_code == 400 and "not allowed" in response.json()["detail"]


def test_gallery_lists_serves_and_deletes(tmp_path):
    web = make_app(tmp_path)
    out = web.settings.output_dir
    (out / "a.png").write_bytes(png_bytes())
    (out / "a.metadata.json").write_text('{"prompt": "teapot", "seed": 3}')
    (out / "a.web.json").write_text('{"command": "mflux-generate-qwen-2.1", "payload": {}}')
    (tmp_path / "secret.png").write_bytes(b"secret")
    client = client_for(web)
    listing = client.get("/api/gallery").json()
    assert listing["total"] == 1
    assert listing["items"][0] | {"modified": 0} == {
        "name": "a.png",
        "modified": 0,
        "command": "mflux-generate-qwen-2.1",
        "prompt": "teapot",
        "seed": 3,
        "reloadable": True,
    }
    assert client.get("/api/images/a.png").content == png_bytes()
    assert client.get("/api/images/a.png/metadata").json()["metadata"]["seed"] == 3
    assert client.get("/api/images/..%2Fsecret.png").status_code in (400, 404)
    assert client.delete("/api/images/a.png", headers=csrf(client)).status_code == 200
    assert not (out / "a.png").exists() and not (out / "a.metadata.json").exists()


def test_commands_endpoint_lists_phase_one_commands(tmp_path):
    client = client_for(make_app(tmp_path))
    names = {c["command"] for c in client.get("/api/commands").json()["commands"]}
    assert {"mflux-generate-qwen-2.1", "mflux-generate-z-image-turbo", "mflux-generate-flux2"} <= names


def test_stale_session_cookie_is_ignored(tmp_path):
    client = client_for(make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY)))
    client.cookies.set(SESSION_COOKIE_NAME, "forged.token.value")
    assert client.get("/api/commands").status_code == 401


def test_clear_history_forgets_finished_jobs_and_their_uploads(tmp_path):
    web = make_app(tmp_path)
    client = client_for(web)
    upload_id = client.post(
        "/api/uploads", files={"file": ("a.png", png_bytes(), "image/png")}, headers=csrf(client)
    ).json()["id"]
    stale_upload = client.post(
        "/api/uploads", files={"file": ("b.png", png_bytes((1, 2, 3)), "image/png")}, headers=csrf(client)
    ).json()["id"]
    body = {
        "command": "mflux-generate-qwen-2.1",
        "options": {"--prompt": "secret prompt"},
        "image": {"upload": upload_id},
    }
    finished = client.post("/api/generate", json=body, headers=csrf(client)).json()
    client.post(f"/api/jobs/{finished['id']}/cancel", headers=csrf(client))
    queued = client.post("/api/generate", json=body, headers=csrf(client)).json()

    result = client.delete("/api/jobs", headers=csrf(client)).json()

    assert result == {"cleared_jobs": 1, "removed_uploads": 1}
    remaining = [job["id"] for job in client.get("/api/jobs").json()["jobs"]]
    assert remaining == [queued["id"]]
    assert (web.settings.upload_dir / upload_id).exists(), "a queued job still needs its init image"
    assert not (web.settings.upload_dir / stale_upload).exists()
    assert client.get(f"/api/jobs/{finished['id']}").status_code == 404


def test_clear_history_needs_csrf(tmp_path):
    client = client_for(make_app(tmp_path))
    assert client.delete("/api/jobs").status_code == 403


def test_quiet_access_log_hides_polling_but_keeps_actions():
    import logging

    from mflux.web.seed.cli import QuietAccessLog

    def record(method, path):
        return logging.LogRecord(
            "uvicorn.access",
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:5", method, path, "1.1", 200),
            None,
        )

    quiet = QuietAccessLog()
    for path in (
        "/api/status",
        "/api/session",
        "/api/jobs",
        "/api/jobs/abc/events",
        "/api/images/a.png",
        "/static/app.css",
    ):
        assert not quiet.filter(record("GET", path)), path
    for method, path in (("POST", "/api/generate"), ("DELETE", "/api/jobs"), ("GET", "/"), ("GET", "/gallery")):
        assert quiet.filter(record(method, path)), path


def test_bearer_guesses_are_throttled(tmp_path):
    client = client_for(make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY)))
    for _ in range(6):
        client.get("/api/status", headers={"Authorization": "Bearer wrong-key-value"})
    # The public session endpoint must not become an unthrottled way to test keys.
    response = client.get("/api/session", headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 429
    assert "Retry-After" in response.headers


def test_session_reports_bearer_authentication(tmp_path):
    client = client_for(make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY)))
    assert client.get("/api/session", headers={"Authorization": f"Bearer {KEY}"}).json()["authenticated"] is True
    assert client.get("/api/session").json()["authenticated"] is False


def test_throttle_is_per_client_behind_a_proxy(tmp_path):
    web = make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY), allowed_hosts=["mflux.example.ts.net"])
    client = client_for(web)
    attacker = {"X-Forwarded-For": "spoofed, 203.0.113.9"}
    for _ in range(6):
        client.post("/api/login", json={"api_key": "wrong-key-value"}, headers={**attacker, **csrf(client)})
    assert client.post("/api/login", json={"api_key": KEY}, headers={**attacker, **csrf(client)}).status_code == 429
    victim = {"X-Forwarded-For": "198.51.100.7"}
    assert client.post("/api/login", json={"api_key": KEY}, headers={**victim, **csrf(client)}).status_code == 200


def test_forwarded_for_is_ignored_without_a_proxy(tmp_path):
    client = client_for(make_app(tmp_path, api_key_hash=WebAuth.hash_key(KEY)))
    for index in range(6):
        headers = {"X-Forwarded-For": f"203.0.113.{index}", **csrf(client)}
        client.post("/api/login", json={"api_key": "wrong-key-value"}, headers=headers)
    assert client.post("/api/login", json={"api_key": KEY}, headers=csrf(client)).status_code == 429


def test_chunked_body_is_refused(tmp_path):
    client = client_for(make_app(tmp_path))

    def body():
        yield b'{"command": "mflux-generate", "options": {"--prompt": "x"}}'

    response = client.post("/api/validate", content=body(), headers=csrf(client))
    assert response.status_code == 411


@pytest.mark.parametrize(
    "extra",
    [
        {"preview_every": "abc"},
        {"preview_every": 2.5},
        {"preview_every": True},
        {"preview_every": 10_000},
        {"model": "dev"},
        {"image": "upload.png"},
        {"command": ["mflux-generate"]},
    ],
)
def test_malformed_generate_payload_is_a_client_error(tmp_path, extra):
    web = make_app(tmp_path)
    client = TestClient(web.app, base_url=BASE, client=("127.0.0.1", 50000), raise_server_exceptions=False)
    body = {"command": "mflux-generate-z-image-turbo", "options": {"--prompt": "x"}, **extra}
    assert client.post("/api/generate", json=body, headers=csrf(client)).status_code == 400


def test_generate_returns_503_when_server_is_stopping(tmp_path):
    web = make_app(tmp_path)
    client = client_for(web)
    web.runner._stopping = True
    body = {"command": "mflux-generate-z-image-turbo", "options": {"--prompt": "x"}}
    response = client.post("/api/generate", json=body, headers=csrf(client))
    assert response.status_code == 503
    assert "stopping" in response.json()["detail"]
