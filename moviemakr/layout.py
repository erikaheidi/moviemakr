"""The run directory, and the host <-> container path model.

The container sees exactly three mounts. `RunLayout` owns both halves of that
model: where every artefact lands on the host, and how a host path is spelled
inside the container.

Methods take a scene `slug`, never a `Scene`, which is what keeps this module a
leaf that the rest of the package can depend on freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

CONTAINER_MODELS = "/models"
CONTAINER_ASSETS = "/assets"
CONTAINER_OUT = "/out"

# sd-cli always writes WebM, whatever container the finished movie uses.
CLIP_SUFFIX = "webm"

RUN_SUBDIRS = ("scenes", "frames", "normalized", "logs")


def slugify(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "scene"


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Paths for one script's run. All three mount bases are pre-resolved."""

    run_dir: Path
    model_root: Path
    assets_dir: Path
    name_slug: str
    container: str

    @classmethod
    def build(cls, *, run_dir: Path, model_root: Path, assets_dir: Path,
              name_slug: str, container: str) -> "RunLayout":
        return cls(
            run_dir=run_dir.resolve(),
            model_root=model_root.resolve(),
            assets_dir=assets_dir.resolve(),
            name_slug=name_slug,
            container=container,
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
        """Raw model output. Always .webm - only the movie follows `container`."""
        return self.scenes_dir / f"{slug}.{CLIP_SUFFIX}"

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
            try:
                rel = host.relative_to(base)
            except ValueError:
                continue
            return mount if rel == Path(".") else f"{mount}/{rel.as_posix()}"
        raise ConfigError(
            f"path is outside every mounted directory and cannot be reached "
            f"from the container: {host}"
        )
