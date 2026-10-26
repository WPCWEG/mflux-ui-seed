import asyncio
import contextlib
import json
import logging
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mflux.web.seed.auth import REMEMBER_ME_MAX_AGE, SESSION_COOKIE_NAME, SESSION_MAX_AGE, LoginThrottle, WebAuth
from mflux.web.seed.i18n import jinja_t, template_context
from mflux.web.seed.invocation import Invocation, InvocationError
from mflux.web.seed.network import NetworkPolicy
from mflux.web.seed.paths import PathGuard, PathRejected
from mflux.web.seed.runner import JobRunner
from mflux.web.seed.schema import FormSchema
from mflux.web.seed.settings import WebSettings

logger = logging.getLogger("mflux.web")

WEB_DIR = Path(__file__).parent
PUBLIC_PATHS = ("/login", "/setup", "/api/login", "/api/setup", "/api/session", "/favicon.svg")
UNSAFE_METHODS = ("POST", "PUT", "PATCH", "DELETE")
FORWARDING_HEADERS = ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip")
MAX_UPLOAD_PIXELS = 50_000_000
MAX_PREVIEW_EVERY = 1000
CSRF_HEADER = "x-mflux-csrf"
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)


class WebApp:
    def __init__(self, settings: WebSettings, runner: JobRunner | None = None):
        self.settings = settings
        self.auth = WebAuth(settings.secret_key, settings.api_key_hash)
        self.throttle = LoginThrottle()
        self.guard = PathGuard(settings.output_dir, settings.upload_dir, settings.models_dirs, settings.lora_dirs)
        self.schema = FormSchema()
        self.runner = runner or JobRunner(
            cache_size=settings.cache_size, idle_unload_seconds=settings.idle_unload_minutes * 60
        )
        self.templates = Jinja2Templates(
            directory=str(WEB_DIR / "templates"),
            context_processors=[template_context],
        )
        self.templates.env.globals["t"] = jinja_t
        self.app = self._build()

    @property
    def auth_required(self) -> bool:
        return self.auth.enabled or self.settings.require_auth

    def _build(self) -> FastAPI:
        app = FastAPI(title="mflux-web", docs_url=None, redoc_url=None, openapi_url=None, lifespan=self._lifespan)
        app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
        app.middleware("http")(self._guard_request)
        self._register_pages(app)
        self._register_auth_api(app)
        self._register_generation_api(app)
        self._register_gallery_api(app)
        return app

    @contextlib.asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.runner.start()
        yield
        self.runner.stop()

    async def _guard_request(self, request: Request, call_next):
        if not NetworkPolicy.host_header_allowed(
            request.headers.get("host"), self.settings.host, self.settings.allowed_hosts
        ):
            return WebApp._secured(JSONResponse({"detail": "Host not allowed"}, status_code=421))
        if WebApp._content_length(request) > (self.settings.max_upload_mb + 1) * 1024 * 1024:
            return WebApp._secured(JSONResponse({"detail": "Request body too large"}, status_code=413))
        if "transfer-encoding" in request.headers and "content-length" not in request.headers:
            # The size check above needs Content-Length. A chunked body would be read whole.
            return WebApp._secured(JSONResponse({"detail": "Content-Length required"}, status_code=411))
        if not self.auth_required and WebApp._proxied(request):
            # An unauthenticated server only trusts callers on this machine. A request that
            # came through a proxy is from somewhere else, even if its socket is loopback.
            return WebApp._secured(
                JSONResponse({"detail": "Proxied access requires an API key (--api-key)"}, status_code=403)
            )
        path = request.url.path
        is_public = path in PUBLIC_PATHS or path.startswith("/static/")
        bearer = WebApp._bearer(request)
        bearer_ok = False
        if bearer is not None and self.auth_required:
            # Every bearer guess costs a PBKDF2 run, so it goes through the login throttle
            # and off the event loop, like a login attempt.
            client = self._throttle_key(request)
            if retry_after := self.throttle.retry_after(client):
                return WebApp._secured(
                    JSONResponse(
                        {"detail": f"Too many failed attempts; retry in {retry_after}s"},
                        status_code=429,
                        headers={"Retry-After": str(retry_after)},
                    )
                )
            bearer_ok = await asyncio.to_thread(self.auth.verify_api_key, bearer)
            if bearer_ok:
                self.throttle.record_success(client)
            else:
                self.throttle.record_failure(client)
                logger.warning("Failed bearer auth from %s (key fingerprint %s)", client, WebAuth.fingerprint(bearer))
            authenticated = bearer_ok
        else:
            authenticated = self._is_authenticated(request)
        request.state.authenticated = authenticated
        if not is_public and not authenticated:
            if request.method == "GET" and not path.startswith("/api/"):
                target = "/setup" if self.settings.require_auth and not self.auth.enabled else "/login"
                return WebApp._secured(RedirectResponse(target, status_code=303))
            return WebApp._secured(JSONResponse({"detail": "Authentication required"}, status_code=401))
        if request.method in UNSAFE_METHODS and not self._csrf_ok(request, bearer_ok):
            return WebApp._secured(JSONResponse({"detail": "CSRF check failed; reload the page"}, status_code=403))
        response = await call_next(request)
        if path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        elif path in ("/", "/gallery", "/login", "/setup"):
            response.headers.setdefault("Cache-Control", "private, no-store")
        return WebApp._secured(response)

    def _is_authenticated(self, request: Request) -> bool:
        if not self.auth_required:
            return True
        return self.auth.verify_session_token(request.cookies.get(SESSION_COOKIE_NAME))

    def _csrf_ok(self, request: Request, bearer_ok: bool) -> bool:
        origin = request.headers.get("origin")
        if origin and origin != "null":
            origin_host = origin.split("://", 1)[-1]
            if origin_host != request.headers.get("host"):
                return False
        if bearer_ok:
            # Scripted clients authenticate per request and never carry a cookie to ride on.
            return True
        return self.auth.verify_csrf(request.cookies.get(SESSION_COOKIE_NAME), request.headers.get(CSRF_HEADER))

    def _register_pages(self, app: FastAPI) -> None:
        @app.get("/")
        def index(request: Request):
            return self.templates.TemplateResponse(request, "index.html", {"page": "generate"})

        @app.get("/gallery")
        def gallery(request: Request):
            return self.templates.TemplateResponse(request, "gallery.html", {"page": "gallery"})

        @app.get("/login")
        def login(request: Request):
            if not self.auth_required:
                return RedirectResponse("/", status_code=303)
            if not self.auth.enabled:
                return RedirectResponse("/setup", status_code=303)
            return self.templates.TemplateResponse(request, "login.html", {"page": "login"})

        @app.get("/setup")
        def setup(request: Request):
            if not self._setup_allowed(request):
                return RedirectResponse("/login" if self.auth.enabled else "/", status_code=303)
            return self.templates.TemplateResponse(request, "setup.html", {"page": "setup"})

        @app.get("/favicon.svg")
        def favicon():
            return FileResponse(WEB_DIR / "static" / "favicon.svg", media_type="image/svg+xml")

    def _register_auth_api(self, app: FastAPI) -> None:
        @app.get("/api/session")
        def session(request: Request):
            return {
                "auth_required": self.auth_required,
                "auth_configured": self.auth.enabled,
                "authenticated": getattr(request.state, "authenticated", False),
                "csrf": self.auth.csrf_token(request.cookies.get(SESSION_COOKIE_NAME)),
            }

        @app.post("/api/login")
        async def login(request: Request):
            client = self._throttle_key(request)
            retry_after = self.throttle.retry_after(client)
            if retry_after:
                raise HTTPException(
                    429, f"Too many failed attempts; retry in {retry_after}s", headers={"Retry-After": str(retry_after)}
                )
            body = await WebApp._json(request)
            api_key = str(body.get("api_key") or "")
            if not self.auth.enabled or not await asyncio.to_thread(self.auth.verify_api_key, api_key):
                self.throttle.record_failure(client)
                logger.warning("Failed login from %s (key fingerprint %s)", client, WebAuth.fingerprint(api_key))
                raise HTTPException(401, "Invalid API key")
            self.throttle.record_success(client)
            return self._session_response(bool(body.get("remember")))

        @app.post("/api/logout")
        def logout():
            response = JSONResponse({"ok": True})
            response.delete_cookie(SESSION_COOKIE_NAME, path="/")
            return response

        @app.post("/api/setup")
        async def setup(request: Request):
            if not self._setup_allowed(request):
                raise HTTPException(403, "First-run key setup is only available over loopback before a key exists")
            body = await WebApp._json(request)
            api_key = str(body.get("api_key") or "")
            if api_key != str(body.get("api_key_confirm") or ""):
                raise HTTPException(400, "Keys do not match")
            if problem := WebAuth.validate_new_key(api_key):
                raise HTTPException(400, problem)
            self.auth.set_api_key(api_key)
            self.settings.api_key_hash = self.auth.api_key_hash
            self.settings.persist()
            logger.info("API key configured through first-run setup")
            return self._session_response(remember=False)

    def _register_generation_api(self, app: FastAPI) -> None:
        @app.get("/api/commands")
        def commands():
            return {
                "commands": self.schema.commands(),
                "local_models": self.guard.list_local_models(),
                "local_loras": self.guard.list_local_loras(),
            }

        @app.get("/api/status")
        def status():
            return self.runner.status()

        @app.post("/api/models/unload")
        def unload_models():
            return {"scheduled": self.runner.request_unload()}

        @app.post("/api/uploads")
        async def upload(file: UploadFile):
            return {"id": await self._store_upload(file)}

        @app.post("/api/validate")
        async def validate(request: Request):
            invocation = self._invocation(await WebApp._json(request))
            try:
                messages = await asyncio.to_thread(invocation.validate)
            except InvocationError as exc:
                raise HTTPException(400, str(exc)) from exc
            return {"shell_command": invocation.shell_command(), "messages": messages}

        @app.post("/api/generate")
        async def generate(request: Request):
            if self.runner.stopping:
                raise HTTPException(503, "Server is stopping; no new jobs accepted")
            payload = await WebApp._json(request)
            preview_every = WebApp._preview_every(payload.get("preview_every"))
            invocation = self._invocation(payload)
            try:
                await asyncio.to_thread(invocation.validate)
            except InvocationError as exc:
                raise HTTPException(400, str(exc)) from exc
            try:
                job = self.runner.submit(invocation, payload, preview_every=preview_every)
            except InvocationError as exc:
                raise HTTPException(503, str(exc)) from exc
            return job.snapshot()

        @app.get("/api/jobs")
        def jobs():
            return {"jobs": [job.snapshot() for job in reversed(self.runner.jobs())]}

        @app.delete("/api/jobs")
        def clear_jobs():
            # Job history only ever lives in this process; clearing it also drops the init
            # images uploaded for those jobs, so nothing of them is left on disk.
            cleared = self.runner.clear_history()
            removed_uploads = self._purge_uploads(keep=self.runner.active_uploads())
            return {"cleared_jobs": cleared, "removed_uploads": removed_uploads}

        @app.get("/api/jobs/{job_id}")
        def job(job_id: str):
            return self._job(job_id).snapshot()

        @app.post("/api/jobs/{job_id}/cancel")
        def cancel(job_id: str):
            self._job(job_id)
            return {"cancelled": self.runner.cancel(job_id)}

        @app.get("/api/jobs/{job_id}/preview.jpg")
        def preview(job_id: str):
            data = self._job(job_id).preview_jpeg
            if data is None:
                raise HTTPException(404, "No preview yet")
            return Response(data, media_type="image/jpeg")

        @app.get("/api/jobs/{job_id}/events")
        async def events(job_id: str, request: Request):
            job = self._job(job_id)

            async def stream():
                seq, last_beat = 0, time.monotonic()
                while True:
                    if await request.is_disconnected():
                        return
                    done = job.finished or self.runner.stopping
                    fresh = job.events_since(seq)
                    for event in fresh:
                        yield f"data: {json.dumps(event)}\n\n"
                    seq += len(fresh)
                    if done and not fresh:
                        return
                    if time.monotonic() - last_beat > 15:
                        last_beat = time.monotonic()
                        yield ": keep-alive\n\n"
                    await asyncio.sleep(0.25)

            return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    def _register_gallery_api(self, app: FastAPI) -> None:
        @app.get("/api/gallery")
        def gallery(offset: int = 0, limit: int = 60):
            limit = min(max(limit, 1), 200)
            offset = max(offset, 0)
            files = sorted(
                (p for p in self.guard.output_dir.glob("*.png") if p.is_file() and not p.is_symlink()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            items = []
            for path in files[offset : offset + limit]:
                sidecar = self._read_sidecar(path, ".web.json")
                metadata = self._read_sidecar(path, ".metadata.json")
                items.append(
                    {
                        "name": path.name,
                        "modified": path.stat().st_mtime,
                        "command": sidecar.get("command") if sidecar else None,
                        "prompt": (metadata or {}).get("prompt")
                        or ((sidecar or {}).get("payload", {}).get("options", {}).get("--prompt")),
                        "seed": (metadata or {}).get("seed"),
                        "reloadable": sidecar is not None,
                    }
                )
            return {"items": items, "total": len(files), "offset": offset}

        @app.get("/api/images/{name}")
        def image(name: str):
            path = self._output_file(name)
            if path.suffix != ".png" or not path.is_file():
                raise HTTPException(404, "Image not found")
            return FileResponse(path, media_type="image/png")

        @app.get("/api/images/{name}/metadata")
        def metadata(name: str):
            path = self._output_file(name)
            if not path.is_file():
                raise HTTPException(404, "Image not found")
            return {
                "metadata": self._read_sidecar(path, ".metadata.json"),
                "web": self._read_sidecar(path, ".web.json"),
            }

        @app.delete("/api/images/{name}")
        def delete(name: str):
            path = self._output_file(name)
            if path.suffix != ".png" or not path.is_file():
                raise HTTPException(404, "Image not found")
            for candidate in (path, path.with_suffix(".metadata.json"), path.with_suffix(".web.json")):
                candidate.unlink(missing_ok=True)
            return {"deleted": name}

    def _invocation(self, payload: dict) -> Invocation:
        stem = self.guard.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{secrets.token_hex(3)}"
        try:
            return Invocation.from_payload(payload, self.guard, self.schema, stem)
        except InvocationError as exc:
            raise HTTPException(400, str(exc)) from exc

    def _job(self, job_id: str):
        job = self.runner.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    def _output_file(self, name: str) -> Path:
        try:
            return self.guard.output_file(name)
        except PathRejected as exc:
            raise HTTPException(400, str(exc)) from exc

    async def _store_upload(self, file: UploadFile) -> str:
        import PIL.Image

        limit = self.settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"Upload exceeds {self.settings.max_upload_mb} MB")

        def _reencode() -> str:
            import io

            try:
                with PIL.Image.open(io.BytesIO(data)) as probe:
                    width, height = probe.size
                    if width * height > MAX_UPLOAD_PIXELS:
                        raise HTTPException(400, f"Image is too large ({width}x{height}); the limit is 50 megapixels")
                    probe.verify()
                with PIL.Image.open(io.BytesIO(data)) as image:
                    image.load()
                    # Re-encoding drops anything that is not pixels before a model reads it.
                    converted = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
            except (PIL.Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
                raise HTTPException(400, f"Not a readable image: {exc}") from exc
            name = f"{secrets.token_hex(8)}.png"
            self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
            converted.save(self.settings.upload_dir / name, format="PNG")
            return name

        return await asyncio.to_thread(_reencode)

    def _purge_uploads(self, keep: set[str]) -> int:
        removed = 0
        upload_dir = self.settings.upload_dir
        if not upload_dir.is_dir():
            return 0
        for path in upload_dir.iterdir():
            if path.is_file() and not path.is_symlink() and path.name not in keep:
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def _setup_allowed(self, request: Request) -> bool:
        return (
            self.settings.require_auth
            and not self.auth.enabled
            and NetworkPolicy.is_loopback_host(self.settings.host)
            and NetworkPolicy.is_loopback_host(WebApp._client(request))
            and not WebApp._proxied(request)
        )

    def _read_sidecar(self, image_path: Path, suffix: str) -> dict | None:
        try:
            path = self.guard.output_file(image_path.with_suffix(suffix).name)
        except PathRejected:
            return None
        return WebApp._read_json(path) if path.is_file() else None

    def _session_response(self, remember: bool) -> JSONResponse:
        token = self.auth.create_session_token(remember=remember)
        response = JSONResponse({"ok": True, "csrf": self.auth.csrf_token(token)})
        response.set_cookie(
            SESSION_COOKIE_NAME,
            token,
            max_age=REMEMBER_ME_MAX_AGE if remember else SESSION_MAX_AGE,
            httponly=True,
            samesite="strict",
            secure=self.settings.secure_cookies,
            path="/",
        )
        return response

    @staticmethod
    def _secured(response: Response) -> Response:
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        return response

    @staticmethod
    def _bearer(request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        return None

    @staticmethod
    def _proxied(request: Request) -> bool:
        return any(header in request.headers for header in FORWARDING_HEADERS)

    @staticmethod
    def _content_length(request: Request) -> int:
        try:
            return int(request.headers.get("content-length") or 0)
        except ValueError:
            return 0

    @staticmethod
    def _client(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _throttle_key(self, request: Request) -> str:
        # Behind a reverse proxy every client has the proxy's socket address, so one attacker
        # would lock out everyone. The proxy appends the address it saw as the last
        # X-Forwarded-For entry. The client writes the entries before it, so they are not trusted.
        client = WebApp._client(request)
        behind_proxy = self.settings.behind_https or any(
            not NetworkPolicy.is_loopback_host(h) for h in self.settings.allowed_hosts
        )
        forwarded = request.headers.get("x-forwarded-for")
        if behind_proxy and forwarded and NetworkPolicy.is_loopback_host(client):
            return forwarded.split(",")[-1].strip() or client
        return client

    @staticmethod
    def _preview_every(value) -> int:
        if value is None or value == "":
            return 0
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_PREVIEW_EVERY:
            raise HTTPException(400, f"preview_every must be a whole number from 0 to {MAX_PREVIEW_EVERY}")
        return value

    @staticmethod
    async def _json(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "Request body must be JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(400, "Request body must be a JSON object")
        return body

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None
