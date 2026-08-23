"""Turning URL-supplied strings into paths that are certainly inside the workspace.

The server hands out files, so this is the one security-relevant module in the
web layer. Every path that originates in a request goes through `safe_path`.
Stdlib only, so it is testable without FastAPI installed.
"""

from __future__ import annotations

from pathlib import Path

from ..layout import slugify


class UnsafePath(ValueError):
    """A request asked for something outside the directory it was scoped to."""


def safe_path(base: Path, relative: str) -> Path:
    """Resolve `relative` under `base`, or raise.

    Rejects absolute paths, `..` escapes, and symlinks pointing out of the tree.
    `base` and the result are both fully resolved before comparison, so a
    workspace reached through a symlink still works.
    """
    if not relative or relative in (".", "/"):
        raise UnsafePath("empty path")

    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise UnsafePath(f"absolute paths are not allowed: {relative}")
    if any(part == ".." for part in candidate.parts):
        raise UnsafePath(f"path escapes the workspace: {relative}")

    base = base.resolve()
    resolved = (base / candidate).resolve()
    if resolved != base and base not in resolved.parents:
        raise UnsafePath(f"path escapes the workspace: {relative}")
    return resolved


def rel_key(base: Path, path: Path) -> str:
    """The URL key for a path inside `base`: a POSIX relative path."""
    return path.resolve().relative_to(base.resolve()).as_posix()


def safe_stem(name: str, *, fallback: str = "untitled") -> str:
    """A filename stem from arbitrary user input: slugified, never empty.

    Reuses the package's own `slugify` so a draft titled "Beach Drive" and a
    scene id "beach-drive" agree on their spelling. Directory separators are
    dropped rather than escaped - this produces a *name*, never a path.
    """
    stem = Path(name.strip()).name
    stem = stem.rsplit(".", 1)[0] if "." in stem[1:] else stem
    if not any(c.isalnum() for c in stem):
        return fallback
    return slugify(stem)
