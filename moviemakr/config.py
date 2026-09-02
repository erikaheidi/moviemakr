"""Load a YAML script into `Script` / `Scene`.

Everything is resolved here: defaults merged with per-scene overrides, every
asset path turned into an absolute host path, every reference image checked for
reachability from inside the container. Missing files fail here, not two hours
into a render.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from .backends import DEFAULT_BACKEND, check_name
from .errors import ConfigError, check_keys
from .layout import RunLayout, Workspace, slugify
from .media import CONTAINERS

TOP_KEYS = frozenset(
    {"name", "backend", "model", "docker", "defaults", "continuity", "output", "scenes"}
)
MODEL_KEYS = frozenset({"root", "diffusion_model", "llm", "vae", "audio_vae"})
MODEL_REQUIRED = ("diffusion_model", "llm", "vae")
DOCKER_KEYS = frozenset({"image", "devices", "run_as_current_user"})
CONTINUITY_KEYS = frozenset({"anchors", "anchor_videos", "chain_from_previous"})
OUTPUT_KEYS = frozenset({"container", "audio", "music", "music_gain_db"})
# Scene keys that are not settings; everything else must be a SceneSettings field.
SCENE_KEYS = frozenset({"id", "prompt", "ref_images", "ref_videos", "chain_from_previous"})

DEFAULT_IMAGE = "ghcr.io/leejet/stable-diffusion.cpp:master-vulkan"
DEFAULT_DEVICES = ["/dev/dri/card1", "/dev/dri/renderD128", "/dev/kfd"]


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def _int(key: str, value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ConfigError(f"{where}: {key} must be an integer, got {value!r}")
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"{where}: {key} must be an integer, got {value!r}") from None


def _positive_int(key: str, value: Any, where: str) -> int:
    number = _int(key, value, where)
    if number < 1:
        raise ConfigError(f"{where}: {key} must be at least 1, got {number}")
    return number


def _opt_positive_int(key: str, value: Any, where: str) -> int | None:
    return None if value is None else _positive_int(key, value, where)


def _float(key: str, value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{where}: {key} must be a number, got {value!r}")
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"{where}: {key} must be a number, got {value!r}") from None


def _str(key: str, value: Any, where: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        raise ConfigError(f"{where}: {key} must be text, got {value!r}")
    return str(value)


def _opt_str(key: str, value: Any, where: str) -> str | None:
    return None if value is None else _str(key, value, where)


def _str_tuple(key: str, value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise ConfigError(f"{where}: {key} must be a list, got {value!r}")
    return tuple(str(item) for item in value)


_COERCERS = {
    "width": _positive_int,
    "height": _positive_int,
    "fps": _positive_int,
    "video_frames": _positive_int,
    "cfg_scale": _float,
    "seed": _int,
    "steps": _opt_positive_int,
    "sampling_method": _opt_str,
    "negative_prompt": _str,
    "style_suffix": _str,
    "extra_args": _str_tuple,
}


@dataclass(frozen=True, slots=True)
class SceneSettings:
    """Render settings for one scene.

    Frozen on purpose: these feed the fingerprint, so a mutation after load would
    desync a scene from its own argv. `slots` additionally turns a typo like
    `settings.video_frame = 90` into an AttributeError rather than a silent no-op.
    """

    width: int = 540
    height: int = 960
    fps: int = 24
    video_frames: int = 120
    cfg_scale: float = 1.0
    seed: int = 42
    steps: int | None = None
    sampling_method: str | None = None
    negative_prompt: str = ""
    style_suffix: str = ""
    # A tuple, not a list: a frozen dataclass must not hand out a shared mutable.
    extra_args: tuple[str, ...] = ()

    FIELDS: ClassVar[frozenset[str]] = frozenset(_COERCERS)

    def merge(self, overrides: Mapping[str, Any], where: str) -> "SceneSettings":
        check_keys(where, overrides, self.FIELDS)
        return dataclasses.replace(
            self,
            **{key: _COERCERS[key](key, value, where) for key, value in overrides.items()},
        )


# --------------------------------------------------------------------------
# dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scene:
    index: int
    id: str
    slug: str
    prompt: str
    settings: SceneSettings
    ref_images: tuple[Path, ...]  # host paths, resolved
    ref_videos: tuple[Path, ...]  # host paths to source video files, resolved
    chain_from_previous: bool

    def full_prompt(self) -> str:
        suffix = (self.settings.style_suffix or "").strip()
        if suffix:
            return f"{self.prompt.rstrip().rstrip('.')}. {suffix}"
        return self.prompt


@dataclass(frozen=True, slots=True)
class DockerConfig:
    image: str
    devices: tuple[str, ...]
    run_as_current_user: bool


@dataclass(frozen=True, slots=True)
class OutputConfig:
    container: str
    audio: str
    music: Path | None
    music_gain_db: float

    @property
    def keep_audio(self) -> bool:
        return self.audio == "keep"


@dataclass(frozen=True, slots=True)
class Script:
    name: str
    path: Path
    workspace: Workspace  # data root the assets and renders came from
    model_files: dict[str, Path]
    docker: DockerConfig
    output: OutputConfig
    scenes: tuple[Scene, ...]
    layout: RunLayout
    # Last, with a default, so constructing a Script positionally keeps working.
    backend: str = DEFAULT_BACKEND

    @property
    def run_dir(self) -> Path:
        return self.layout.run_dir

    @property
    def model_root(self) -> Path:
        return self.layout.model_root

    @property
    def assets_dir(self) -> Path:
        return self.layout.assets_dir

    @property
    def primary_size(self) -> tuple[int, int]:
        """Scene 1's dimensions decide the movie; others are padded to fit."""
        first = self.scenes[0].settings
        return first.width, first.height

    @property
    def fps(self) -> int:
        return self.scenes[0].settings.fps


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _as_dict(raw: Any, key: str) -> dict:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{key}: must be a mapping, got {value!r}")
    return value


def _resolve_file(rel: str, what: str, assets_dir: Path) -> Path:
    """Resolve a script-relative or absolute path, requiring it to exist."""
    path = Path(str(rel)).expanduser()
    host = (path if path.is_absolute() else (assets_dir / path)).resolve()
    if not host.is_file():
        raise ConfigError(f"{what} not found: {host}")
    return host


def _resolve_ref_image(rel: str, what: str, layout: RunLayout) -> Path:
    """A reference image must also be reachable from inside the container.

    Reference *videos* are exempt: they are transcoded into frame directories
    under the run dir, so their sources can live anywhere.
    """
    host = _resolve_file(rel, what, layout.assets_dir)
    try:
        layout.to_container(host)
    except ConfigError:
        raise ConfigError(
            f"{what} is outside every mounted directory, so the container cannot "
            f"read it: {host}\n"
            f"  Reference images must live under {layout.assets_dir}"
        ) from None
    return host


def load_script(script_path: Path, workspace: Workspace) -> Script:
    if not script_path.is_file():
        raise ConfigError(f"script not found: {script_path}")
    with script_path.open() as fh:
        # A syntax error has to become a ConfigError like every other bad
        # script: callers only guard that one type, and an escaping YAMLError
        # is a traceback on the CLI and a 500 that takes the web index down.
        try:
            raw = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {script_path}: {exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"script must be a mapping at the top level: {script_path}")
    check_keys("top level", raw, TOP_KEYS)

    name = raw.get("name") or script_path.stem
    backend = check_name(str(raw.get("backend") or DEFAULT_BACKEND))

    # --- model ---
    model = _as_dict(raw, "model")
    check_keys("model", model, MODEL_KEYS)
    model_root = Path(str(model.get("root", ""))).expanduser()
    if not model_root.is_dir():
        raise ConfigError(f"model.root is not a directory: {model_root}")

    model_files: dict[str, Path] = {}
    for key in (*MODEL_REQUIRED, "audio_vae"):
        rel = model.get(key)
        if rel is None:
            if key in MODEL_REQUIRED:
                raise ConfigError(f"model.{key} is required")
            continue
        host = (model_root / str(rel)).resolve()
        if not host.is_file():
            raise ConfigError(f"model.{key} not found: {host}")
        model_files[key] = host

    # --- docker ---
    docker_raw = _as_dict(raw, "docker")
    check_keys("docker", docker_raw, DOCKER_KEYS)
    docker = DockerConfig(
        image=str(docker_raw.get("image", DEFAULT_IMAGE)),
        devices=tuple(str(d) for d in (docker_raw.get("devices") or DEFAULT_DEVICES)),
        run_as_current_user=bool(docker_raw.get("run_as_current_user", True)),
    )

    # --- output ---
    output_raw = _as_dict(raw, "output")
    check_keys("output", output_raw, OUTPUT_KEYS)
    container = str(output_raw.get("container", "mp4"))
    if container not in CONTAINERS:
        raise ConfigError(f"output.container must be one of {sorted(CONTAINERS)}")
    audio = str(output_raw.get("audio", "keep"))
    if audio not in ("keep", "strip"):
        raise ConfigError("output.audio must be 'keep' or 'strip'")

    # --- layout, built before scenes so refs can be checked against the mounts ---
    assets_dir = workspace.assets_dir
    layout = RunLayout.build(
        run_dir=workspace.renders_dir / slugify(name),
        model_root=model_root,
        assets_dir=assets_dir,
        name_slug=slugify(name),
        container=container,
    )

    music_raw = output_raw.get("music")
    output = OutputConfig(
        container=container,
        audio=audio,
        music=_resolve_file(music_raw, "output.music", assets_dir) if music_raw else None,
        music_gain_db=_float("music_gain_db", output_raw.get("music_gain_db", -18), "output"),
    )

    # --- defaults + continuity ---
    defaults_raw = _as_dict(raw, "defaults")
    script_defaults = SceneSettings().merge(defaults_raw, "defaults")

    continuity = _as_dict(raw, "continuity")
    check_keys("continuity", continuity, CONTINUITY_KEYS)
    anchor_paths = [
        _resolve_ref_image(a, "continuity anchor", layout)
        for a in (continuity.get("anchors") or [])
    ]
    anchor_video_paths = [
        _resolve_file(a, "continuity anchor_video", assets_dir)
        for a in (continuity.get("anchor_videos") or [])
    ]
    global_chain = bool(continuity.get("chain_from_previous", False))

    # --- scenes ---
    raw_scenes = raw.get("scenes") or []
    if not raw_scenes:
        raise ConfigError("script has no scenes")

    scenes: list[Scene] = []
    seen_ids: set[str] = set()
    allowed_scene_keys = SCENE_KEYS | SceneSettings.FIELDS

    for i, rs in enumerate(raw_scenes, start=1):
        if not isinstance(rs, dict) or not rs.get("prompt"):
            raise ConfigError(f"scene {i}: 'prompt' is required")
        sid = str(rs.get("id") or f"scene{i}")
        if sid in seen_ids:
            raise ConfigError(f"duplicate scene id: {sid}")
        seen_ids.add(sid)

        where = f"scene {i} ('{sid}')"
        check_keys(where, rs, allowed_scene_keys)
        overrides = {k: v for k, v in rs.items() if k not in SCENE_KEYS}
        settings = script_defaults.merge(overrides, where)

        refs = list(anchor_paths)
        for rel in rs.get("ref_images") or []:
            refs.append(_resolve_ref_image(rel, f"{where} ref_image", layout))

        ref_videos = list(anchor_video_paths)
        for rel in rs.get("ref_videos") or []:
            ref_videos.append(_resolve_file(rel, f"{where} ref_video", assets_dir))

        scenes.append(
            Scene(
                index=i,
                id=sid,
                slug=f"{i:03d}-{slugify(sid)}",
                prompt=str(rs["prompt"]),
                settings=settings,
                ref_images=tuple(refs),
                ref_videos=tuple(ref_videos),
                chain_from_previous=bool(rs.get("chain_from_previous", global_chain)),
            )
        )

    return Script(
        name=name,
        path=script_path,
        workspace=workspace,
        model_files=model_files,
        docker=docker,
        output=output,
        scenes=tuple(scenes),
        layout=layout,
        backend=backend,
    )
