"""Render backends - one module per engine that can turn a Scene into a clip.

Each backend owns its own command/graph construction *and* its own fingerprint,
because a fingerprint is by definition "everything that decides what this engine
will produce". Two engines cannot share one hash: the same scene rendered through
stable-diffusion.cpp and through ComfyUI is not the same output, and a run that
switched engines must re-render rather than silently resume.

- `sdcpp` runs stable-diffusion.cpp in a container, one `docker run` per scene.

The registry maps the script's `backend:` key to a module. It is kept here rather
than in `config` so that loading a script never imports an engine it will not use.
"""

from __future__ import annotations

from ..errors import ConfigError, suggest

BACKENDS = ("sdcpp", "comfy")
DEFAULT_BACKEND = "sdcpp"


def check_name(name: str) -> str:
    """Validate a `backend:` value at load time, with a did-you-mean hint.

    Mirrors `check_keys`: a typo must fail while loading, not two hours into a
    render, and not by silently falling back to the default engine.
    """
    if name in BACKENDS:
        return name
    hint = suggest(name, BACKENDS)
    raise ConfigError(
        f"backend: unknown backend {name!r}"
        + (f"   (did you mean '{hint}'?)" if hint else "")
        + f"\n  allowed: {', '.join(BACKENDS)}"
    )


def resolve(name: str):
    """Import and return the backend module for `name`."""
    check_name(name)
    if name == "sdcpp":
        from . import sdcpp

        return sdcpp
    if name == "comfy":
        from . import comfy

        return comfy
    raise AssertionError(f"backend {name!r} is registered but not wired up")
