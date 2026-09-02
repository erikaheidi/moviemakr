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


@pytest.fixture(autouse=True)
def _no_ambient_workspace(monkeypatch):
    """A developer's own $MOVIEMAKR_WORKSPACE must not leak into the suite."""
    from moviemakr.layout import WORKSPACE_ENV

    monkeypatch.delenv(WORKSPACE_ENV, raising=False)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """A synthetic workspace root: the data dirs, with no code checkout nearby."""
    root = tmp_path / "project"
    (root / "assets").mkdir(parents=True)
    return root


@pytest.fixture
def workspace(project_root: Path):
    from moviemakr.layout import Workspace

    return Workspace.at(project_root)


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


# ComfyUI-side model *names*, not paths: the server resolves them in its own
# models directory. Fixed, because they appear in the golden graph and its hash.
COMFY_MODELS = {
    "diffusion_model": "minimax_h3_fl2va_pruned_bf16.safetensors",
    "text_encoder": "qwen3vl_32b_minimax_h3_bf16.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}


@pytest.fixture
def comfy_script(base_script: dict) -> dict:
    """A minimal comfy-backend script: no model/docker block, no refs."""
    script = copy.deepcopy(base_script)
    script.pop("model")
    script.pop("docker")
    script["backend"] = "comfy"
    script["comfy"] = dict(COMFY_MODELS)
    return script


@pytest.fixture
def write_comfy(project_root: Path, comfy_script: dict):
    """Write a comfy-backend script and return its path."""

    def _write(overrides: dict | None = None, filename: str = "comfy.yaml") -> Path:
        data = deep_merge(comfy_script, overrides or {})
        path = project_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return _write


@pytest.fixture
def load_comfy(write_comfy, workspace):
    from moviemakr.config import load_script

    def _load(overrides: dict | None = None):
        return load_script(write_comfy(overrides), workspace)

    return _load


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
def web_workspace(tmp_path, model_root, base_script):
    """A populated workspace for the web tests.

    Nested scripts, one that cannot load, a draft, two assets, and one script
    with a rendered clip plus a finished movie. Still hermetic: the "media" is
    plain bytes and the clip carries a stored probe in state.json, so nothing
    here reaches ffprobe.
    """
    import json

    from moviemakr.layout import Workspace

    root = tmp_path / "ws"
    for sub in ("scripts/h3", "assets", "drafts"):
        (root / sub).mkdir(parents=True)

    (root / "assets" / "josy-reference.jpg").write_bytes(b"anchor-bytes")
    (root / "assets" / "unused.png").write_bytes(b"unused-bytes")
    (root / "drafts" / "picnic.md").write_text(
        "# Beach picnic\n\nJosy finds a sandwich. Ends on a wave.\n")

    def script(name, anchors):
        return deep_merge(base_script, {
            "name": name,
            "continuity": {"anchors": anchors, "chain_from_previous": True},
            "scenes": [
                {"id": "opening", "prompt": "Scene one.", "chain_from_previous": False},
                {"id": "middle", "prompt": "Scene two."},
            ],
        })

    (root / "scripts" / "simple.yaml").write_text(
        yaml.safe_dump(script("simple", []), sort_keys=False))
    (root / "scripts" / "h3" / "beach.yaml").write_text(
        yaml.safe_dump(script("beach drive", ["josy-reference.jpg"]), sort_keys=False))
    # Mirrors scripts/h3/josy-house-party.yaml: an anchor that is not in assets/.
    (root / "scripts" / "h3" / "broken.yaml").write_text(
        yaml.safe_dump(script("broken", ["does-not-exist.jpg"]), sort_keys=False))

    run_dir = root / "renders" / "simple"
    (run_dir / "scenes").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "scenes" / "001-opening.webm").write_bytes(b"clip-bytes")
    (run_dir / "simple.mp4").write_bytes(b"movie-bytes")
    (run_dir / "logs" / "001-opening.attempt1.log").write_text("rendering\ndone\n")
    # A stored probe keeps `scene_table` away from ffprobe.
    (run_dir / "state.json").write_text(json.dumps({
        "scenes": {
            "opening": {
                "state": "rendered",
                "elapsed": 902.5,
                "probe": {"frames": 90, "duration": 3.75, "has_audio": True,
                          "width": 540, "height": 960},
            }
        }
    }))

    return Workspace.at(root)


@pytest.fixture
def load(write_script, workspace):
    """Write a script and load it, returning the Script object."""
    import moviemakr as M

    def _load(overrides: dict | None = None, filename: str = "test.yaml"):
        return M.load_script(_resolve(write_script(overrides, filename)), workspace)

    def _resolve(path: Path) -> Path:
        return path.resolve()

    return _load
