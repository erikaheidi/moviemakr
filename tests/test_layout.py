"""RunLayout: where artefacts land, and how host paths are spelled in the container."""

from __future__ import annotations

from pathlib import Path

import pytest

from moviemakr.errors import ConfigError
from moviemakr.layout import RunLayout, slugify


# --------------------------------------------------------------------------
# slugify
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("kitchen", "kitchen"),
        ("Kitchen", "kitchen"),
        ("too spicy", "too-spicy"),
        ("too  spicy", "too-spicy"),
        ("--leading-and-trailing--", "leading-and-trailing"),
        ("a/b\\c:d", "a-b-c-d"),
        ("", "scene"),
        ("!!!", "scene"),
        ("cats-cooking-h3", "cats-cooking-h3"),
    ],
)
def test_slugify(text, expected):
    assert slugify(text) == expected


def test_slugify_is_idempotent():
    for text in ("Too Spicy!", "a  b", "---"):
        once = slugify(text)
        assert slugify(once) == once


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def test_clip_is_always_webm(layout):
    """sd-cli writes WebM; only the movie follows output.container."""
    assert layout.container == "mp4"
    assert layout.clip("001-opening").name == "001-opening.webm"


def test_normalized_follows_container(layout):
    assert layout.normalized("001-opening").name == "001-opening.mp4"

    webm = RunLayout.build(
        run_dir=layout.run_dir, model_root=layout.model_root,
        assets_dir=layout.assets_dir, name_slug="x", container="webm",
    )
    assert webm.normalized("001-opening").name == "001-opening.webm"


def test_per_scene_paths(layout):
    run = layout.run_dir
    assert layout.clip("001-a") == run / "scenes" / "001-a.webm"
    assert layout.frame("001-a") == run / "frames" / "001-a.last.png"
    assert layout.log("001-a", 2) == run / "logs" / "001-a.attempt2.log"
    assert layout.normalized("001-a") == run / "normalized" / "001-a.mp4"


def test_run_level_paths(layout):
    run = layout.run_dir
    assert layout.state_file == run / "state.json"
    assert layout.concat_file == run / "concat.txt"
    assert layout.movie == run / "test-movie.mp4"
    assert layout.concat_tmp == run / ".concat-tmp.mp4"


def test_refvideo_dir_tag(layout):
    got = layout.refvideo_dir(Path("/somewhere/Cat Cooking.webm"), 544, 960)
    assert got == layout.run_dir / "refvideos" / "cat-cooking-544x960"


def test_ensure_dirs_creates_exactly_four(layout):
    layout.ensure_dirs()
    made = sorted(p.name for p in layout.run_dir.iterdir() if p.is_dir())
    assert made == ["frames", "logs", "normalized", "scenes"]


def test_ensure_dirs_is_idempotent(layout):
    layout.ensure_dirs()
    layout.ensure_dirs()
    assert (layout.run_dir / "scenes").is_dir()


def test_layout_is_frozen(layout):
    with pytest.raises(Exception):
        layout.container = "webm"


# --------------------------------------------------------------------------
# to_container
# --------------------------------------------------------------------------


def test_to_container_each_mount(layout):
    assert layout.to_container(layout.model_root / "h3/x.gguf") == "/models/h3/x.gguf"
    assert layout.to_container(layout.assets_dir / "anchor.png") == "/assets/anchor.png"
    assert layout.to_container(layout.run_dir / "scenes/a.webm") == "/out/scenes/a.webm"


def test_to_container_nested(layout):
    deep = layout.assets_dir / "a" / "b" / "c.png"
    assert layout.to_container(deep) == "/assets/a/b/c.png"


def test_to_container_mount_root_has_no_trailing_slash(layout):
    assert layout.to_container(layout.assets_dir) == "/assets"


def test_to_container_normalizes_dot_segments(layout):
    messy = layout.assets_dir / "sub" / ".." / "anchor.png"
    assert layout.to_container(messy) == "/assets/anchor.png"


def test_to_container_rejects_outside_paths(layout, tmp_path):
    stray = tmp_path / "elsewhere" / "x.png"
    with pytest.raises(ConfigError) as exc:
        layout.to_container(stray)
    assert str(stray) in str(exc.value)


def test_to_container_precedence_models_before_assets(tmp_path):
    """First match wins, in the order models, assets, run dir."""
    root = tmp_path / "shared"
    (root / "renders" / "m").mkdir(parents=True)
    nested = RunLayout.build(
        run_dir=root / "renders" / "m",
        model_root=root,  # the run dir sits *inside* the model root
        assets_dir=tmp_path / "assets",
        name_slug="m",
        container="mp4",
    )
    (tmp_path / "assets").mkdir(exist_ok=True)
    # models is checked first, so the shared path resolves against /models.
    assert nested.to_container(root / "renders" / "m" / "x.webm") == "/models/renders/m/x.webm"
