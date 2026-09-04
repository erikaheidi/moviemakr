"""A read-mostly web view of a moviemakr workspace.

Browse scripts and their per-scene state, play and download finished movies,
write drafts, and upload reference images - from a phone or a laptop, over
Tailscale, instead of sftp.

Rendering is deliberately not here: it stays on the CLI over ssh. The app never
starts a container and never calls an LLM.

`paths`, `browse` and `assets` are pure logic and import nothing beyond the
stdlib and the core package, so they stay testable where FastAPI is not
installed. Only `app` needs the web extra, which is why the imports below are
deferred.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..layout import WORKSPACE_ENV, Workspace

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

__all__ = ["create_app", "missing_modules", "run_server"]

# What the `web` extra installs, by import name. Because every import of these
# is deferred, importing this module always succeeds - so a caller cannot learn
# the extra is absent by catching ImportError around the import, and has to ask.
# `starlette` is omitted: fastapi depends on it, so it is never the one missing.
REQUIRED_MODULES = ("fastapi", "uvicorn", "jinja2", "multipart")


def missing_modules() -> list[str]:
    """Which of the extra's modules are not importable. Empty means ready."""
    from importlib.util import find_spec

    missing = []
    for name in REQUIRED_MODULES:
        try:
            found = find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def create_app(workspace: Workspace) -> FastAPI:
    from .app import create_app as _create_app

    return _create_app(workspace)


def app_from_env() -> FastAPI:
    """Entry point for uvicorn's reloader, which needs an import string."""
    return create_app(Workspace.resolve())


def run_server(workspace: Workspace, *, host: str = "127.0.0.1", port: int = 8765,
               reload: bool = False) -> int:
    import uvicorn

    print(f"workspace: {workspace.root}")
    print(f"serving on http://{host}:{port}")

    if reload:
        # The reloader re-imports in a fresh process, so the workspace has to
        # travel through the environment rather than a closure.
        os.environ[WORKSPACE_ENV] = str(workspace.root)
        uvicorn.run("moviemakr.web:app_from_env", factory=True, reload=True,
                    host=host, port=port, log_level="info")
    else:
        uvicorn.run(create_app(workspace), host=host, port=port, log_level="info")
    return 0
