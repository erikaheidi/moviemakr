"""Errors and key validation.

A YAML typo used to be free: an unrecognized key was dropped in silence, and you
found out that `video_frame: 90` did nothing only after the render finished.
`check_keys` turns that into a load-time failure with a suggestion.
"""

from __future__ import annotations

import difflib
from collections.abc import Collection, Mapping
from typing import Any


class ConfigError(Exception):
    """A script is malformed, or refers to something that is not there."""


def suggest(word: str, options: Collection[str], *, cutoff: float = 0.6) -> str | None:
    """Closest allowed key to `word`, or None when nothing is close enough.

    `options` is sorted first so ties resolve the same way every run - otherwise
    a test asserting the suggestion text is flaky.
    """
    match = difflib.get_close_matches(word, sorted(options), n=1, cutoff=cutoff)
    return match[0] if match else None


def check_keys(where: str, data: Mapping[str, Any], allowed: Collection[str]) -> None:
    """Raise unless every key in `data` is allowed.

    `where` names the block for the message, e.g. "defaults" or "scene 2 ('kitchen')".
    """
    unknown = sorted(set(data) - set(allowed))
    if not unknown:
        return

    lines = [f"{where}: unknown key(s)"]
    for key in unknown:
        hint = suggest(key, allowed)
        lines.append(f"  {key}" + (f"   (did you mean '{hint}'?)" if hint else ""))
    lines.append(f"  allowed: {', '.join(sorted(allowed))}")
    raise ConfigError("\n".join(lines))
