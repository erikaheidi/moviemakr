"""Shared fixtures.

Everything here is hermetic: no Docker, no GPU, no ffmpeg. That is possible
because `sd_args` maps every host path through `to_container` before it reaches
the command line, so the argv - and therefore the fingerprint - contains only
container-side paths and never the tmp_path a test happens to run in.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Relative to the model root; the names are arbitrary but must stay fixed,
# because they appear in the argv that the golden fingerprints hash.
MODEL_FILES = {
    "diffusion_model": "h3/diffusion.gguf",
    "llm": "h3/llm.gguf",
    "vae": "h3/video_vae.safetensors",
    "audio_vae": "h3/audio_vae.safetensors",
}


def deep_merge(base: dict, overrides: dict) -> dict:
    """Merge `overrides` into a copy of `base`, recursing into nested dicts.

    A None value deletes the key, so a test can drop `audio_vae` or `steps`.
    """
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if value is _DELETE:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class _Delete:
    def __repr__(self) -> str:
        return "<DELETE>"


_DELETE = _Delete()


@pytest.fixture
def DELETE():
    """Sentinel for removing a key in a `write_script` override."""
    return _DELETE


@pytest.fixture
def model_root(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    for rel in MODEL_FILES.values():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "assets").mkdir(parents=True)
    return root


@pytest.fixture
def assets_dir(project_root: Path) -> Path:
    return project_root / "assets"


@pytest.fixture
def make_asset(assets_dir: Path):
    """Create a file under assets/ with known bytes and return its path."""

    def _make(name: str, content: bytes = b"stub-image-bytes") -> Path:
        path = assets_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    return _make


@pytest.fixture
def layout(tmp_path: Path, model_root: Path, project_root: Path):
    """A bare RunLayout, with no Script behind it."""
    from moviemakr.layout import RunLayout

    return RunLayout.build(
        run_dir=project_root / "renders" / "test-movie",
        model_root=model_root,
        assets_dir=project_root / "assets",
        name_slug="test-movie",
        container="mp4",
    )


@pytest.fixture
def base_script(model_root: Path) -> dict:
    return {
        "name": "test-movie",
        "model": {"root": str(model_root), **MODEL_FILES},
        "docker": {
            "image": "test/image:tag",
            "devices": [],
            "run_as_current_user": True,
        },
        "defaults": {
            "width": 540,
            "height": 960,
            "fps": 24,
            "video_frames": 120,
            "cfg_scale": 1.0,
            "seed": 42,
            "negative_prompt": "",
            "style_suffix": "",
            "extra_args": [],
        },
        "continuity": {"anchors": [], "chain_from_previous": False},
        "output": {
            "container": "mp4",
            "audio": "keep",
            "music": None,
            "music_gain_db": -18,
        },
        "scenes": [{"id": "opening", "prompt": "A test scene."}],
    }


@pytest.fixture
def write_script(project_root: Path, base_script: dict):
    """Write a YAML script into the project and return its path.

    Overrides are deep-merged into `base_script`, so a test can change one
    nested key without restating the rest.
    """

    def _write(overrides: dict | None = None, filename: str = "test.yaml") -> Path:
        data = deep_merge(base_script, overrides or {})
        path = project_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return _write


@pytest.fixture
def load(write_script, project_root: Path):
    """Write a script and load it, returning the Script object."""
    import moviemakr as M

    def _load(overrides: dict | None = None, filename: str = "test.yaml"):
        return M.load_script(_resolve(write_script(overrides, filename)), project_root)

    def _resolve(path: Path) -> Path:
        return path.resolve()

    return _load
