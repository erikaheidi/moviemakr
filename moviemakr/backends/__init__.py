"""Render backends - one module per engine that can turn a Scene into a clip.

Each backend owns its own command construction *and* its own fingerprint,
because a fingerprint is by definition "everything that decides what this engine
will produce".

- `sdcpp` runs stable-diffusion.cpp in a container, one `docker run` per scene.
"""

from __future__ import annotations
