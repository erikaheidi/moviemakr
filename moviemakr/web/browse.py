"""Reading the workspace into plain data for the templates.

No FastAPI here: every function takes a `Workspace` and returns dicts and
dataclasses, so the whole browse layer is testable without a web server. A
script that fails to load is reported as a row with an `error`, never raised -
one broken YAML must not take the index page down.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Script, load_script
from ..errors import ConfigError
from ..layout import Workspace
from ..report import fmt_duration
from ..status import scene_rows
from .paths import rel_key, safe_path

DRAFT_SUFFIX = ".md"
SCRIPT_SUFFIXES = (".yaml", ".yml")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _mtime(path: Path) -> dt.datetime | None:
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


# --------------------------------------------------------------------------
# scripts
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ScriptRow:
    """One entry on the index. `error` set means the YAML would not load."""

    key: str                      # path relative to scripts/, the URL key
    name: str
    path: Path
    modified: dt.datetime | None
    error: str | None = None
    scene_count: int = 0
    movie: Path | None = None
    movie_size: int = 0
    rendered: int = 0
    stale: int = 0
    pending: int = 0

    @property
    def has_movie(self) -> bool:
        return self.movie is not None

    @property
    def progress(self) -> str:
        if self.error:
            return "invalid"
        if not self.scene_count:
            return "-"
        return f"{self.rendered}/{self.scene_count}"


def script_files(workspace: Workspace) -> list[Path]:
    """Every script in the workspace, recursively - they nest under h3/, short/."""
    scripts_dir = workspace.scripts_dir
    if not scripts_dir.is_dir():
        return []
    found = [
        p for p in scripts_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SCRIPT_SUFFIXES
    ]
    return sorted(found, key=lambda p: rel_key(scripts_dir, p))


def load_script_at(workspace: Workspace, key: str) -> Script:
    """Load one script by its URL key. Raises ConfigError or UnsafePath."""
    return load_script(safe_path(workspace.scripts_dir, key), workspace)


def script_row(workspace: Workspace, path: Path, *, with_state: bool = False) -> ScriptRow:
    """Summarise one script. `with_state` recomputes fingerprints - not free."""
    key = rel_key(workspace.scripts_dir, path)
    row = ScriptRow(key=key, name=path.stem, path=path, modified=_mtime(path))

    try:
        script = load_script(path, workspace)
    except ConfigError as exc:
        row.error = str(exc)
        return row

    row.name = script.name
    row.scene_count = len(script.scenes)

    movie = script.layout.movie
    if movie.is_file():
        row.movie = movie
        row.movie_size = _size(movie)

    if with_state:
        for entry in scene_rows(script):
            state = entry["state"]
            if state == "rendered":
                row.rendered += 1
            elif state == "stale":
                row.stale += 1
            else:
                row.pending += 1
    else:
        # Cheap approximation for the index: a clip on disk counts as rendered.
        # The script page recomputes fingerprints and can disagree by design.
        row.rendered = sum(
            1 for s in script.scenes
            if script.layout.clip(s.slug).is_file()
            and _size(script.layout.clip(s.slug)) > 0
        )
        row.pending = row.scene_count - row.rendered

    return row


def script_rows(workspace: Workspace) -> list[ScriptRow]:
    return [script_row(workspace, p) for p in script_files(workspace)]


def script_folders(workspace: Workspace) -> list[str]:
    """Subdirectories of scripts/ that already hold scripts, as POSIX keys.

    Offered to the upload form so a script uploaded from another machine keeps
    landing in the same folder it lives in there, rather than at the top level.
    """
    scripts_dir = workspace.scripts_dir
    folders = {
        rel_key(scripts_dir, p.parent)
        for p in script_files(workspace)
        if p.parent != scripts_dir
    }
    return sorted(folders)


# --------------------------------------------------------------------------
# one script's scenes
# --------------------------------------------------------------------------


def scene_table(script: Script) -> list[dict[str, Any]]:
    """`scene_rows` plus the presentation bits the template needs."""
    layout = script.layout
    table = []
    for row in scene_rows(script):
        scene = row["scene"]
        clip = layout.clip(scene.slug)
        probe = row.get("probe") or {}
        frame = layout.frame(scene.slug)
        table.append({
            "index": scene.index,
            "id": scene.id,
            "slug": scene.slug,
            "state": row["state"],
            "duration": fmt_duration(probe.get("duration")),
            "elapsed": fmt_duration(row.get("elapsed")),
            "size": probe.get("width") and f"{probe['width']}x{probe['height']}" or "-",
            "clip": clip if clip.is_file() else None,
            "clip_size": human_size(_size(clip)) if clip.is_file() else "",
            "frame": frame if frame.is_file() else None,
            "prompt": scene.prompt,
            "refs": [r.name for r in scene.ref_images],
            "chained": scene.chain_from_previous,
        })
    return table


def log_files(script: Script) -> list[dict[str, Any]]:
    logs_dir = script.layout.logs_dir
    if not logs_dir.is_dir():
        return []
    files = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {"name": p.name, "size": human_size(_size(p)), "modified": _mtime(p)}
        for p in files
    ]


# --------------------------------------------------------------------------
# drafts
# --------------------------------------------------------------------------


@dataclass(slots=True)
class DraftRow:
    slug: str
    title: str
    path: Path
    modified: dt.datetime | None
    excerpt: str = ""


def draft_title(text: str, slug: str) -> str:
    """First markdown heading, else first non-empty line, else the slug."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.lstrip("#").strip() or slug
    return slug


def draft_rows(workspace: Workspace) -> list[DraftRow]:
    drafts_dir = workspace.drafts_dir
    if not drafts_dir.is_dir():
        return []
    rows = []
    for path in sorted(drafts_dir.glob(f"*{DRAFT_SUFFIX}")):
        text = path.read_text(errors="replace")
        body = "\n".join(text.splitlines()[1:]).strip()
        rows.append(DraftRow(
            slug=path.stem,
            title=draft_title(text, path.stem),
            path=path,
            modified=_mtime(path),
            excerpt=body[:180] + ("…" if len(body) > 180 else ""),
        ))
    rows.sort(key=lambda r: r.modified or dt.datetime.min, reverse=True)
    return rows


# --------------------------------------------------------------------------
# assets
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AssetRow:
    name: str
    path: Path
    size: int
    modified: dt.datetime | None
    used_by: list[str] = field(default_factory=list)

    @property
    def size_h(self) -> str:
        return human_size(self.size)


def asset_files(workspace: Workspace) -> list[Path]:
    assets_dir = workspace.assets_dir
    if not assets_dir.is_dir():
        return []
    return sorted(
        (p for p in assets_dir.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name.lower(),
    )


def asset_rows(workspace: Workspace, *, usage: dict[str, list[str]] | None = None
               ) -> list[AssetRow]:
    usage = usage or {}
    return [
        AssetRow(name=p.name, path=p, size=_size(p), modified=_mtime(p),
                 used_by=usage.get(p.name, []))
        for p in asset_files(workspace)
    ]


def asset_usage(workspace: Workspace) -> dict[str, list[str]]:
    """Which scripts mention each asset, by plain text search.

    Deliberately textual rather than load-based: a script that fails to load
    still tells you its asset is in use, and an unloadable script is exactly
    when you want to know.
    """
    usage: dict[str, list[str]] = {}
    names = [p.name for p in asset_files(workspace)]
    if not names:
        return usage
    for path in script_files(workspace):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        key = rel_key(workspace.scripts_dir, path)
        for name in names:
            if name in text:
                usage.setdefault(name, []).append(key)
    return usage


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------


def workspace_summary(workspace: Workspace) -> dict[str, Any]:
    rows = script_rows(workspace)
    movies = [r for r in rows if r.has_movie]
    assets = asset_files(workspace)
    return {
        "root": workspace.root,
        "scripts": rows,
        "folders": script_folders(workspace),
        "drafts": draft_rows(workspace),
        "script_count": len(rows),
        "invalid_count": sum(1 for r in rows if r.error),
        "movie_count": len(movies),
        "movie_size": human_size(sum(r.movie_size for r in movies)),
        "asset_count": len(assets),
        "asset_size": human_size(sum(_size(p) for p in assets)),
    }
