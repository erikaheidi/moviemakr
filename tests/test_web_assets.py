"""Upload handling and thumbnails.

ffmpeg is stubbed out - the suite is hermetic, and what matters here is the
decision logic: what is accepted, what it gets named, and when it is resized.
"""

from __future__ import annotations

import pytest

from moviemakr.web import assets as A


@pytest.fixture
def assets_dir(tmp_path):
    path = tmp_path / "assets"
    path.mkdir()
    return path


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Report a size for any staged file and make resizing a plain copy."""
    calls = []

    def _size(path):
        return getattr(_size, "value", (4000, 3000))

    def _run(cmd):
        calls.append(cmd)
        # ffmpeg's last argument is the destination.
        dest = cmd[-1]
        with open(dest, "wb") as fh:
            fh.write(b"resized")
        return None

    monkeypatch.setattr(A, "image_size", _size)
    monkeypatch.setattr(A, "_run", _run)
    _size.calls = calls
    return _size


# --- what is accepted ------------------------------------------------------


@pytest.mark.parametrize("name", ["a.jpg", "a.JPG", "a.jpeg", "a.png", "a.webp"])
def test_accepted_suffixes(assets_dir, fake_ffmpeg, name):
    dest, _ = A.store_upload(assets_dir, name, b"data")
    assert dest.is_file()


def test_jpeg_is_normalised_to_jpg(assets_dir, fake_ffmpeg):
    dest, _ = A.store_upload(assets_dir, "photo.jpeg", b"data")
    assert dest.suffix == ".jpg"


@pytest.mark.parametrize("name,expected", [
    ("notes.txt", "only "),
    ("clip.mp4", "only "),
    ("photo.heic", "libheif"),
    ("noextension", "only "),
])
def test_rejected_suffixes(assets_dir, fake_ffmpeg, name, expected):
    with pytest.raises(A.UploadRejected, match=expected):
        A.store_upload(assets_dir, name, b"data")


def test_empty_upload_rejected(assets_dir, fake_ffmpeg):
    with pytest.raises(A.UploadRejected, match="empty file"):
        A.store_upload(assets_dir, "a.jpg", b"")


def test_oversized_upload_rejected(assets_dir, fake_ffmpeg, monkeypatch):
    monkeypatch.setattr(A, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(A.UploadRejected, match="exceeds"):
        A.store_upload(assets_dir, "a.jpg", b"x" * 11)


def test_undecodable_bytes_rejected(assets_dir, monkeypatch):
    """The extension is not trusted: ffprobe has to recognise the content."""
    monkeypatch.setattr(A, "image_size", lambda path: None)
    with pytest.raises(A.UploadRejected, match="not a decodable image"):
        A.store_upload(assets_dir, "evil.jpg", b"this is not an image")


def test_rejection_leaves_no_staging_file(assets_dir, monkeypatch):
    monkeypatch.setattr(A, "image_size", lambda path: None)
    with pytest.raises(A.UploadRejected):
        A.store_upload(assets_dir, "evil.jpg", b"nope")
    assert list(assets_dir.iterdir()) == []


# --- naming ----------------------------------------------------------------


def test_filename_is_slugified(assets_dir, fake_ffmpeg):
    dest, _ = A.store_upload(assets_dir, "Josy At The Beach.JPG", b"data")
    assert dest.name == "josy-at-the-beach.jpg"


def test_never_overwrites_an_existing_asset(assets_dir, fake_ffmpeg):
    """A script may already reference the old file by name."""
    (assets_dir / "josy.jpg").write_bytes(b"original")
    first, _ = A.store_upload(assets_dir, "josy.jpg", b"new")
    second, _ = A.store_upload(assets_dir, "josy.jpg", b"newer")
    assert first.name == "josy-2.jpg"
    assert second.name == "josy-3.jpg"
    assert (assets_dir / "josy.jpg").read_bytes() == b"original"


# --- resizing --------------------------------------------------------------


def test_large_images_are_downscaled(assets_dir, fake_ffmpeg):
    fake_ffmpeg.value = (4000, 3000)
    _, note = A.store_upload(assets_dir, "big.jpg", b"data")
    assert fake_ffmpeg.calls, "ffmpeg was not invoked"
    assert "→" in note
    assert str(A.MAX_EDGE) in " ".join(fake_ffmpeg.calls[0])


def test_small_images_are_stored_as_is(assets_dir, fake_ffmpeg):
    fake_ffmpeg.value = (640, 480)
    dest, note = A.store_upload(assets_dir, "small.jpg", b"original-bytes")
    assert not fake_ffmpeg.calls, "small image should not be re-encoded"
    assert note == "640x480"
    assert dest.read_bytes() == b"original-bytes"


def test_keeping_full_size_skips_the_resize(assets_dir, fake_ffmpeg):
    fake_ffmpeg.value = (4000, 3000)
    dest, note = A.store_upload(assets_dir, "big.jpg", b"original-bytes",
                                downscale=False)
    assert not fake_ffmpeg.calls
    assert dest.read_bytes() == b"original-bytes"
    assert note == "4000x3000"


def test_fit_filter_never_upscales():
    """`min(edge, iw)` is what stops a 640px image being blown up to 1280."""
    assert f"min({A.MAX_EDGE},iw)" in A._fit_filter(A.MAX_EDGE)
    assert "force_original_aspect_ratio=decrease" in A._fit_filter(A.MAX_EDGE)


# --- thumbnails ------------------------------------------------------------


def test_thumbnail_is_generated_then_cached(assets_dir, tmp_path, fake_ffmpeg):
    source = assets_dir / "a.jpg"
    source.write_bytes(b"image")
    cache = tmp_path / ".cache"

    first = A.ensure_thumb(cache, source)
    assert first is not None and first.is_file()
    assert len(fake_ffmpeg.calls) == 1

    second = A.ensure_thumb(cache, source)
    assert second == first
    assert len(fake_ffmpeg.calls) == 1, "cached thumbnail was regenerated"


def test_thumbnail_regenerates_when_the_source_changes(assets_dir, tmp_path,
                                                       fake_ffmpeg):
    import os

    source = assets_dir / "a.jpg"
    source.write_bytes(b"image")
    cache = tmp_path / ".cache"
    thumb = A.ensure_thumb(cache, source)

    # Age the thumbnail so the source is unambiguously newer.
    old = thumb.stat().st_mtime - 60
    os.utime(thumb, (old, old))
    A.ensure_thumb(cache, source)
    assert len(fake_ffmpeg.calls) == 2


def test_missing_source_gives_no_thumbnail(tmp_path, fake_ffmpeg):
    assert A.ensure_thumb(tmp_path / ".cache", tmp_path / "gone.jpg") is None


def test_ffmpeg_failure_gives_no_thumbnail(assets_dir, tmp_path, monkeypatch):
    import subprocess

    source = assets_dir / "a.jpg"
    source.write_bytes(b"image")

    def boom(cmd):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(A, "_run", boom)
    assert A.ensure_thumb(tmp_path / ".cache", source) is None
