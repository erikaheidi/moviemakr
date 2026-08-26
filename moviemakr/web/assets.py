"""Taking an uploaded image into the workspace, and thumbnailing for the grid.

Reference images *must* live under `assets/` or `RunLayout.to_container` cannot
map them, so this is what makes drafting from a phone possible at all.

ffmpeg does the pixel work - it is already a hard dependency of the project, and
this avoids adding Pillow. Nothing here imports FastAPI.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..media import ffprobe_json
from .paths import safe_stem

# What a phone will actually hand us. HEIC is excluded on purpose: decoding it
# needs an ffmpeg built with libheif, which is not safe to assume.
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUFFIX_ALIASES = {".jpeg": ".jpg"}

# Phone photos are 3-12 MB and the workspace is a git repo.
MAX_EDGE = 1280
THUMB_EDGE = 320
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


class UploadRejected(ValueError):
    """The upload is not something we are willing to put in the workspace."""


def _fit_filter(edge: int) -> str:
    """Fit inside `edge` on the long side, never upscaling, keeping aspect."""
    return (
        f"scale='min({edge},iw)':'min({edge},ih)'"
        f":force_original_aspect_ratio=decrease:force_divisible_by=2"
    )


def normalize_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise UploadRejected(
            f"{filename}: only {allowed} are accepted "
            f"(HEIC needs an ffmpeg built with libheif)"
        )
    return SUFFIX_ALIASES.get(suffix, suffix)


def unique_path(assets_dir: Path, stem: str, suffix: str) -> Path:
    """Never overwrite an existing asset - a script may already reference it."""
    candidate = assets_dir / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = assets_dir / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def image_size(path: Path) -> tuple[int, int] | None:
    streams = (ffprobe_json(path).get("streams") or [])
    for stream in streams:
        if stream.get("width") and stream.get("height"):
            return int(stream["width"]), int(stream["height"])
    return None


def store_upload(assets_dir: Path, filename: str, data: bytes, *,
                 downscale: bool = True) -> tuple[Path, str]:
    """Write an uploaded image into assets/. Returns (path, human note).

    The bytes are validated by decoding them, not by trusting the extension:
    ffprobe has to recognise the file as an image or it does not land.
    """
    if not data:
        raise UploadRejected(f"{filename}: empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"{filename}: {len(data) / 1024 / 1024:.0f} MB exceeds the "
            f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB limit"
        )

    suffix = normalize_suffix(filename)
    stem = safe_stem(filename, fallback="upload")
    assets_dir.mkdir(parents=True, exist_ok=True)

    staging = assets_dir / f".incoming-{stem}{suffix}"
    staging.write_bytes(data)
    try:
        size = image_size(staging)
        if size is None:
            raise UploadRejected(f"{filename}: not a decodable image")

        dest = unique_path(assets_dir, stem, suffix)
        width, height = size
        if downscale and max(width, height) > MAX_EDGE:
            _run(["ffmpeg", "-v", "error", "-y", "-i", str(staging),
                  "-vf", _fit_filter(MAX_EDGE), "-q:v", "3", str(dest)])
            new = image_size(dest) or (0, 0)
            note = f"{width}x{height} → {new[0]}x{new[1]}"
        else:
            shutil.move(str(staging), dest)
            note = f"{width}x{height}"
        return dest, note
    finally:
        staging.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# thumbnails
# --------------------------------------------------------------------------


def thumb_path(cache_dir: Path, source: Path) -> Path:
    return cache_dir / "thumbs" / f"{source.stem}{source.suffix.lower()}"


def ensure_thumb(cache_dir: Path, source: Path) -> Path | None:
    """Return a cached thumbnail, generating it if stale. None if ffmpeg fails.

    The grid is served over Tailscale to a phone; 20 full-size jpgs is 12 MB of
    mobile data for a page that only needs to show what is there.
    """
    if not source.is_file():
        return None

    dest = thumb_path(cache_dir, source)
    try:
        if dest.is_file() and dest.stat().st_mtime >= source.stat().st_mtime:
            return dest
    except OSError:
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(source),
              "-vf", _fit_filter(THUMB_EDGE), "-q:v", "6", str(dest)])
    except (subprocess.CalledProcessError, OSError):
        return None
    return dest if dest.is_file() else None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)
