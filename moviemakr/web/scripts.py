"""Taking an uploaded YAML script into the workspace.

The mirror of `assets.py` for the other untracked half of a workspace: scripts
live in the workspace, not in this checkout, so a script written on another
machine has to get here somehow. This is that somehow.

Two tiers of checking, deliberately:

- *Rejected* means the bytes are not a moviemakr script at all - not UTF-8, not
  YAML, not a mapping, no `scenes`. Nothing is written.
- *Stored with a warning* means it parses but `load_script` refuses it, almost
  always because a reference image has not been uploaded yet. Order matters
  when you are working from a phone, so the file lands and the problem is
  reported; the index already renders an unloadable script as an error row.

Nothing here imports FastAPI.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..config import load_script
from ..errors import ConfigError
from ..layout import Workspace
from .paths import UnsafePath, safe_path, safe_stem

ALLOWED_SUFFIXES = {".yaml", ".yml"}

# A script is prose and numbers; the largest one here is a few KB.
MAX_UPLOAD_BYTES = 1024 * 1024


class ScriptRejected(ValueError):
    """The upload is not something we are willing to put in scripts/."""


def normalize_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise ScriptRejected(f"{filename}: only {allowed} are accepted")
    return suffix


def decode(filename: str, data: bytes) -> str:
    """Bytes to text, with CRLF normalised - the workspace is a git repo."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ScriptRejected(f"{filename}: not UTF-8 text") from None
    return text.replace("\r\n", "\n")


def check_shape(filename: str, text: str) -> None:
    """Reject anything that is not recognisably a script, before writing it.

    Full validation is `load_script`'s job and happens after the file lands -
    this only catches "that was the wrong file".
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScriptRejected(f"{filename}: invalid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ScriptRejected(f"{filename}: not a mapping at the top level")
    if not isinstance(raw.get("scenes"), list) or not raw["scenes"]:
        raise ScriptRejected(f"{filename}: no scenes - is this a moviemakr script?")


def destination(scripts_dir: Path, filename: str, folder: str = "") -> Path:
    """Where `filename` would land, under an optional subdirectory.

    Scripts nest freely (`h3/`, `short/`), so the folder is part of the form.
    Both halves go through `safe_path`: the folder is typed by hand and the
    filename comes from another machine.
    """
    suffix = normalize_suffix(filename)
    stem = safe_stem(filename, fallback="script")
    # Only whitespace and a trailing slash are tidied away: a leading one has
    # to survive so `safe_path` sees an absolute path and refuses it.
    folder = folder.strip().rstrip("/")
    relative = f"{folder}/{stem}{suffix}" if folder else f"{stem}{suffix}"
    try:
        return safe_path(scripts_dir, relative)
    except UnsafePath as exc:
        raise ScriptRejected(f"{filename}: {exc}") from None


def store_upload(scripts_dir: Path, filename: str, data: bytes, *,
                 folder: str = "", replace: bool = False) -> tuple[Path, str]:
    """Write an uploaded script into scripts/. Returns (path, human note).

    Unlike an asset, a re-upload of the same name is the *expected* case - it is
    how a finalised script gets updated from the other laptop - but it is still
    destructive, so it takes an explicit `replace`.
    """
    if not data:
        raise ScriptRejected(f"{filename}: empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ScriptRejected(
            f"{filename}: {len(data) / 1024:.0f} KB exceeds the "
            f"{MAX_UPLOAD_BYTES // 1024} KB limit"
        )

    text = decode(filename, data)
    check_shape(filename, text)
    dest = destination(scripts_dir, filename, folder)

    if dest.exists() and not replace:
        raise ScriptRejected(
            f"{dest.name} already exists - tick 'replace' to overwrite it"
        )
    existed = dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return dest, "replaced" if existed else "new"


def load_error(path: Path, workspace: Workspace) -> str | None:
    """None if the stored script loads, else the first line of why it does not."""
    try:
        load_script(path, workspace)
    except ConfigError as exc:
        return str(exc).splitlines()[0]
    return None
