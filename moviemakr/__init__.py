"""moviemakr - render a multi-scene movie script with stable-diffusion.cpp in Docker.

Each scene is one `docker run` of sd-cli in vid_gen mode. Scenes are rendered in
order, resumed via fingerprint, retried on failure, then assembled with ffmpeg.

The names re-exported here are the stable surface: they keep working when the
implementation moves between submodules.
"""

from __future__ import annotations

from .assemble import assemble
from .config import (
    DockerConfig,
    OutputConfig,
    Scene,
    SceneSettings,
    Script,
    load_script,
)
from .backends.sdcpp import (
    check_gpu,
    container_name,
    device_gids,
    docker_argv,
    fingerprint,
    format_argv,
    kill_container,
    sd_args,
)
from .errors import ConfigError, check_keys, suggest
from .layout import RunLayout, Workspace, slugify
from .media import CONTAINERS, NormalizeSpec, extract_last_frame, probe_clip
from .render import RenderOptions, SceneJob, render, select_scenes
from .report import fmt_duration, print_summary
from .state import load_state, save_state
from .status import scene_rows

__version__ = "0.2.0"

__all__ = [
    "CONTAINERS",
    "ConfigError",
    "DockerConfig",
    "NormalizeSpec",
    "OutputConfig",
    "RenderOptions",
    "RunLayout",
    "Scene",
    "SceneJob",
    "SceneSettings",
    "Script",
    "Workspace",
    "__version__",
    "assemble",
    "check_gpu",
    "check_keys",
    "container_name",
    "device_gids",
    "docker_argv",
    "extract_last_frame",
    "fingerprint",
    "fmt_duration",
    "format_argv",
    "kill_container",
    "load_script",
    "load_state",
    "print_summary",
    "probe_clip",
    "render",
    "save_state",
    "scene_rows",
    "sd_args",
    "select_scenes",
    "slugify",
    "suggest",
]
