"""The workspace, the run directory, and the host <-> container path model.

`Workspace` is the data root: the directory holding `scripts/`, `assets/`,
`drafts/` and `renders/`. It is deliberately separate from the code checkout so
the package can be installed anywhere and two instances can run against
different content.

The container sees exactly three mounts. `RunLayout` owns both halves of that
model: where every artefact lands on the host, and how a host path is spelled
inside the container.

Methods take a scene `slug`, never a `Scene`, which is what keeps this module a
leaf that the rest of the package can depend on freely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

CONTAINER_MODELS = "/models"
CONTAINER_ASSETS = "/assets"
CONTAINER_OUT = "/out"

WORKSPACE_ENV = "MOVIEMAKR_WORKSPACE"

# sd-cli always writes WebM, whatever container the finished movie uses.
# ComfyUI's SaveVideo writes MP4, so the suffix is per-backend: naming an MP4
# `.webm` still plays (ffmpeg sniffs content) but makes the web view serve it as
# video/webm, and lies to anyone reading the directory.
CLIP_SUFFIX = "webm"

RUN_SUBDIRS = ("scenes", "frames", "normalized", "logs")


def slugify(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "scene"


@dataclass(frozen=True, slots=True)
class Workspace:
    """The data root: scripts, assets, drafts and renders.

    Always separate from the code checkout, so the package can be installed
    anywhere and several instances can run against different content.
    `RunLayout` still takes its three mount bases explicitly - this type only
    decides where they are.
    """

    root: Path

    @classmethod
    def at(cls, root: Path) -> "Workspace":
        return cls(root=Path(root).expanduser().resolve())

    @classmethod
    def resolve(cls, explicit: Path | None = None) -> "Workspace":
        """Pick the workspace: the explicit argument, else $MOVIEMAKR_WORKSPACE.

        There is deliberately no third fallback. The checkout used to serve as
        one, which meant a mistyped or forgotten workspace resolved to a valid
        directory and failed later, confusingly - or worse, quietly wrote a new
        `assets/` into the code tree.
        """
        chosen = explicit
        if chosen is None:
            from_env = os.environ.get(WORKSPACE_ENV)
            chosen = Path(from_env) if from_env else None
        if chosen is None:
            raise ConfigError(
                f"no workspace: pass --workspace or set {WORKSPACE_ENV}"
            )
        workspace = cls.at(chosen)
        if not workspace.root.is_dir():
            raise ConfigError(f"workspace is not a directory: {workspace.root}")
        return workspace

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def drafts_dir(self) -> Path:
        return self.root / "drafts"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache"


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Paths for one script's run. All three mount bases are pre-resolved."""

    run_dir: Path
    # None when the backend has no per-scene container to mount models into -
    # ComfyUI is a long-running server that already owns its own models.
    model_root: Path | None
    assets_dir: Path
    name_slug: str
    container: str
    clip_suffix: str = CLIP_SUFFIX

    @classmethod
    def build(cls, *, run_dir: Path, model_root: Path | None, assets_dir: Path,
              name_slug: str, container: str,
              clip_suffix: str = CLIP_SUFFIX) -> "RunLayout":
        return cls(
            run_dir=run_dir.resolve(),
            model_root=model_root.resolve() if model_root is not None else None,
            assets_dir=assets_dir.resolve(),
            name_slug=name_slug,
            container=container,
            clip_suffix=clip_suffix,
        )

    # --- directories ------------------------------------------------------

    @property
    def scenes_dir(self) -> Path:
        return self.run_dir / "scenes"

    @property
    def frames_dir(self) -> Path:
        return self.run_dir / "frames"

    @property
    def normalized_dir(self) -> Path:
        return self.run_dir / "normalized"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def refvideos_dir(self) -> Path:
        return self.run_dir / "refvideos"

    # --- per-scene files --------------------------------------------------

    def clip(self, slug: str) -> Path:
        """Raw engine output, in whatever the engine writes - not `container`."""
        return self.scenes_dir / f"{slug}.{self.clip_suffix}"

    def frame(self, slug: str) -> Path:
        """Last frame of the scene, fed to the next one when chaining."""
        return self.frames_dir / f"{slug}.last.png"

    def log(self, slug: str, attempt: int) -> Path:
        return self.logs_dir / f"{slug}.attempt{attempt}.log"

    def normalized(self, slug: str) -> Path:
        return self.normalized_dir / f"{slug}.{self.container}"

    def refvideo_dir(self, src: Path, width: int, height: int) -> Path:
        return self.refvideos_dir / f"{slugify(src.stem)}-{width}x{height}"

    # --- run-level files --------------------------------------------------

    @property
    def state_file(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def concat_file(self) -> Path:
        return self.run_dir / "concat.txt"

    @property
    def movie(self) -> Path:
        return self.run_dir / f"{self.name_slug}.{self.container}"

    @property
    def concat_tmp(self) -> Path:
        return self.run_dir / f".concat-tmp.{self.container}"

    # --- operations -------------------------------------------------------

    def ensure_dirs(self) -> None:
        for sub in RUN_SUBDIRS:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)

    def to_container(self, host: Path) -> str:
        """Map a host path to its container-side equivalent.

        Precedence is models, then assets, then the run dir; first match wins.
        Anything outside all three is unreachable from the container.
        """
        host = Path(host).resolve()
        for base, mount in (
            (self.model_root, CONTAINER_MODELS),
            (self.assets_dir, CONTAINER_ASSETS),
            (self.run_dir, CONTAINER_OUT),
        ):
            if base is None:
                continue
            try:
                rel = host.relative_to(base)
            except ValueError:
                continue
            return mount if rel == Path(".") else f"{mount}/{rel.as_posix()}"
        raise ConfigError(
            f"path is outside every mounted directory and cannot be reached "
            f"from the container: {host}"
        )
