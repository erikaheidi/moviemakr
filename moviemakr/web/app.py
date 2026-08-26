"""The HTTP surface. The only module that imports FastAPI.

Kept thin on purpose: every route resolves a path, calls into `browse` /
`assets`, and renders a template. The logic those two modules hold is testable
without a web server, and this file is what you read to see the URL map.

Nothing here starts a render. The app is read-only toward rendering and
read-write only for drafts and for script/asset uploads.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import Script, load_script
from ..errors import ConfigError
from ..layout import Workspace
from . import assets as A
from . import browse as B
from . import scripts as S
from .paths import UnsafePath, rel_key, safe_path, safe_stem

HERE = Path(__file__).resolve().parent

# A response the phone should not cache: state.json changes under an
# ssh-launched render, and a stale scene table is worse than none.
NO_STORE = {"Cache-Control": "no-store"}


def create_app(workspace: Workspace) -> FastAPI:
    app = FastAPI(title="moviemakr", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals["human_size"] = B.human_size
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def page(request: Request, name: str, **context):
        return templates.TemplateResponse(
            request=request, name=name,
            context={"workspace": workspace, **context},
        )

    def script_path(key: str) -> Path:
        try:
            path = safe_path(workspace.scripts_dir, key)
        except UnsafePath as exc:
            raise HTTPException(400, str(exc)) from None
        if not path.is_file():
            raise HTTPException(404, f"no such script: {key}")
        return path

    def get_script(key: str) -> Script:
        """Load a script by URL key. Absent is 404; unloadable is 422."""
        path = script_path(key)
        try:
            return load_script(path, workspace)
        except ConfigError as exc:
            raise HTTPException(422, str(exc)) from None

    def under(base: Path, name: str) -> Path:
        """A single path component supplied by the URL, validated against `base`."""
        try:
            return safe_path(base, name)
        except UnsafePath as exc:
            raise HTTPException(400, str(exc)) from None

    def send(path: Path, *, download: bool, media_type: str | None = None) -> FileResponse:
        """FileResponse handles HTTP Range, which iOS needs to scrub a video."""
        if not path.is_file():
            raise HTTPException(404, f"not found: {path.name}")
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name if download else None,
        )

    def back_to(url: str, **params: str) -> RedirectResponse:
        """Redirect carrying flash text. Filenames and error text go into a
        Location header, so they have to be percent-encoded - a raw space
        there is an invalid header."""
        query = urlencode({k: v for k, v in params.items() if v})
        return RedirectResponse(f"{url}?{query}" if query else url, status_code=303)

    # ----------------------------------------------------------------- index

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, uploaded: str = "", error: str = "", warning: str = ""):
        return page(request, "index.html", uploaded=uploaded, error=error,
                    warning=warning, **B.workspace_summary(workspace))

    # --------------------------------------------------------------- scripts
    #
    # Route order matters: Starlette compiles `{key:path}` to a greedy `.*`, so
    # the longer patterns must be registered first or `/scripts/a/raw` is read
    # as the script named "a/raw".

    @app.post("/scripts")
    async def script_upload(files: list[UploadFile], folder: str = Form(""),
                            replace: bool = Form(False)):
        """Take a finalised script from another machine into the workspace.

        A stored script that will not load is a warning, not a rejection: refs
        are uploaded separately, so the script often arrives first.
        """
        done, warnings, errors = [], [], []
        for upload in files:
            try:
                dest, note = S.store_upload(
                    workspace.scripts_dir, upload.filename or "script.yaml",
                    await upload.read(), folder=folder, replace=replace,
                )
            except S.ScriptRejected as exc:
                errors.append(str(exc))
                continue
            key = rel_key(workspace.scripts_dir, dest)
            done.append(f"{key} ({note})")
            problem = S.load_error(dest, workspace)
            if problem:
                warnings.append(f"{key}: {problem}")
        return back_to(
            "/",
            uploaded=", ".join(done),
            warning="; ".join(warnings),
            error="; ".join(errors),
        )

    @app.get("/scripts/{key:path}/raw", response_class=PlainTextResponse)
    def script_raw(key: str, download: bool = False):
        path = script_path(key)
        if download:
            return send(path, download=True, media_type="text/plain")
        return PlainTextResponse(path.read_text(errors="replace"))

    @app.get("/scripts/{key:path}/scenes", response_class=HTMLResponse)
    def script_scenes(request: Request, key: str):
        """The polled fragment: just the scene table."""
        script = get_script(key)
        response = page(request, "_scenes.html", key=key, script=script,
                        scenes=B.scene_table(script))
        response.headers.update(NO_STORE)
        return response

    @app.get("/scripts/{key:path}/movie")
    def script_movie(key: str, download: bool = False):
        return send(get_script(key).layout.movie, download=download)

    @app.get("/scripts/{key:path}/clip/{slug}")
    def script_clip(key: str, slug: str, download: bool = False):
        layout = get_script(key).layout
        return send(under(layout.scenes_dir, slug), download=download)

    @app.get("/scripts/{key:path}/frame/{name}")
    def script_frame(key: str, name: str):
        layout = get_script(key).layout
        return send(under(layout.frames_dir, name), download=False,
                    media_type="image/png")

    @app.get("/scripts/{key:path}/log/{name}", response_class=PlainTextResponse)
    def script_log(key: str, name: str):
        path = under(get_script(key).layout.logs_dir, name)
        if not path.is_file():
            raise HTTPException(404, f"no such log: {name}")
        return PlainTextResponse(path.read_text(errors="replace"), headers=NO_STORE)

    @app.get("/scripts/{key:path}", response_class=HTMLResponse)
    def script_detail(request: Request, key: str):
        script = get_script(key)
        movie = script.layout.movie
        return page(
            request, "script.html",
            key=key,
            script=script,
            scenes=B.scene_table(script),
            logs=B.log_files(script),
            movie=movie if movie.is_file() else None,
            movie_size=B.human_size(movie.stat().st_size) if movie.is_file() else "",
        )

    # ---------------------------------------------------------------- drafts

    @app.get("/drafts", response_class=HTMLResponse)
    def drafts_index(request: Request):
        return page(request, "drafts.html", drafts=B.draft_rows(workspace))

    @app.post("/drafts")
    def draft_create(title: str = Form("")):
        slug = safe_stem(title, fallback="untitled")
        path = under(workspace.drafts_dir, f"{slug}{B.DRAFT_SUFFIX}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {title.strip() or slug}\n\n")
        return RedirectResponse(f"/drafts/{quote(slug)}", status_code=303)

    @app.get("/drafts/{slug}", response_class=HTMLResponse)
    def draft_detail(request: Request, slug: str, saved: bool = False):
        path = under(workspace.drafts_dir, f"{slug}{B.DRAFT_SUFFIX}")
        if not path.is_file():
            raise HTTPException(404, f"no such draft: {slug}")
        return page(request, "draft.html", slug=slug,
                    text=path.read_text(errors="replace"), saved=saved)

    @app.post("/drafts/{slug}")
    def draft_save(slug: str, text: str = Form("")):
        path = under(workspace.drafts_dir, f"{slug}{B.DRAFT_SUFFIX}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Browsers submit CRLF; the workspace is a git repo, so normalise.
        path.write_text(text.replace("\r\n", "\n"))
        return RedirectResponse(f"/drafts/{quote(slug)}?saved=1", status_code=303)

    @app.post("/drafts/{slug}/delete")
    def draft_delete(slug: str):
        under(workspace.drafts_dir, f"{slug}{B.DRAFT_SUFFIX}").unlink(missing_ok=True)
        return RedirectResponse("/drafts", status_code=303)

    # ---------------------------------------------------------------- assets

    @app.get("/assets", response_class=HTMLResponse)
    def assets_index(request: Request, uploaded: str = "", error: str = ""):
        rows = B.asset_rows(workspace, usage=B.asset_usage(workspace))
        return page(request, "assets.html", assets=rows,
                    uploaded=uploaded, error=error)

    @app.post("/assets")
    async def asset_upload(files: list[UploadFile], keep_full_size: bool = Form(False)):
        done, errors = [], []
        for upload in files:
            try:
                dest, note = A.store_upload(
                    workspace.assets_dir, upload.filename or "upload",
                    await upload.read(), downscale=not keep_full_size,
                )
                done.append(f"{dest.name} ({note})")
            except A.UploadRejected as exc:
                errors.append(str(exc))
        return back_to("/assets", uploaded=", ".join(done), error="; ".join(errors))

    @app.get("/assets/file/{name}")
    def asset_file(name: str, download: bool = False):
        return send(under(workspace.assets_dir, name), download=download)

    @app.get("/assets/thumb/{name}")
    def asset_thumb(name: str):
        path = under(workspace.assets_dir, name)
        thumb = A.ensure_thumb(workspace.cache_dir, path)
        # An undecodable file still gets a response: the full file, which the
        # browser can refuse on its own terms.
        return send(thumb or path, download=False)

    return app
