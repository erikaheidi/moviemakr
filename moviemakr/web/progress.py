"""Progress of a render this process is not running.

`moviemakr serve` never starts a render — rendering stays on the CLI, usually
over ssh — so the web view has to infer progress from what a render leaves on
disk. Two artefacts carry it: `state.json`, for the scenes that finished, and
the newest file in `logs/`, for the one happening now. Nothing here reaches out
to Docker or to ComfyUI: a page render must not hang because the render host is
busy or gone.

Both backends leave a usable trail in the log, and each needs its own line:

    sdcpp   ``  |=========>   | 7/20 - 175.60s/it``   the sampler's own bar
    comfy   ``  [3/10] still rendering (12m32s)``     the poll loop's heartbeat

so an sdcpp scene is carried by its sampling step count and a comfy one by the
elapsed time the heartbeat already prints. Neither is the whole scene — sdcpp
still has to decode the latents afterwards, and the heartbeat knows nothing
about how far along ComfyUI is — so what this module reports is always a floor,
and the templates say "~" everywhere it is shown.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Script

# How long a log may go untouched and still count as a live render. Generous on
# purpose: sdcpp says nothing while it loads 45GB of weights, and again while
# the VAE decodes, and an indicator that blinks off mid-render is worse than one
# that lingers for a few minutes after the run ends.
ACTIVE_SECONDS = 300.0

# The lines that matter are the last few. A log is ~20KB and there is no reason
# to read all of it on every poll.
TAIL_BYTES = 4096

LOG_NAME = re.compile(r"^(?P<slug>.+)\.attempt(?P<attempt>\d+)\.log$")

# stable-diffusion.cpp prints "s/it" once a step is slower than a second, which
# for video it always is — but it flips to "it/s" for the fast passes.
STEP_LINE = re.compile(
    r"\|\s*(?P<step>\d+)/(?P<steps>\d+)\s+-\s+(?P<rate>[\d.]+)(?P<unit>s/it|it/s)"
)
BEAT_LINE = re.compile(r"still rendering \((?:(?P<min>\d+)m)?(?P<sec>\d+)s\)")


@dataclass(slots=True)
class Active:
    """The scene a render appears to be working on right now.

    `elapsed` is measured (comfy prints it); `step`/`steps` are the current
    pass, not the scene. A scene is several passes, so `step == steps` means a
    pass just finished, never that the scene is about to be written.
    """

    index: int
    id: str
    slug: str
    log_name: str
    attempt: int
    quiet: float                        # seconds since the log was last written
    elapsed: float | None = None        # comfy's heartbeat, when there is one
    step: int | None = None
    steps: int | None = None
    seconds_per_step: float | None = None

    @property
    def pass_percent(self) -> int | None:
        if self.step is None or not self.steps:
            return None
        return min(100, round(100 * self.step / self.steps))

    @property
    def pass_remaining(self) -> float | None:
        """Seconds left in the current pass — *not* in the scene."""
        if self.step is None or not self.steps or not self.seconds_per_step:
            return None
        return max(0.0, (self.steps - self.step) * self.seconds_per_step)


@dataclass(slots=True)
class RunProgress:
    """One script's scenes, counted by state, plus whatever is in flight."""

    total: int = 0
    rendered: int = 0
    stale: int = 0
    failed: int = 0
    pending: int = 0
    active: Active | None = None
    mean_elapsed: float | None = None   # per rendered scene, from state.json

    @property
    def running(self) -> bool:
        return self.active is not None

    @property
    def outstanding(self) -> int:
        """Scenes a render would still have to do. Stale counts: it redoes those."""
        return self.stale + self.failed + self.pending

    @property
    def percent(self) -> int:
        if not self.total:
            return 0
        return round(100 * self.rendered / self.total)

    @property
    def slice_percent(self) -> float:
        """How wide one scene is on the bar - the width of the in-flight sliver.

        The sliver marks *which* scene is running, not how far into it the
        render is: the log's step count belongs to one pass of several, so
        filling the sliver in proportion to it would overstate every time.
        """
        return round(100 / self.total, 2) if self.total else 0

    @property
    def eta(self) -> float | None:
        """Rough seconds to a finished movie, from the mean scene so far.

        Only ever an order of magnitude: scenes differ in length, and the
        estimate assumes the render is still running, which nothing here can
        actually know.
        """
        if self.mean_elapsed is None or not self.outstanding:
            return None
        total = self.mean_elapsed * self.outstanding
        if self.active is not None:
            # The scene in flight is part done. Prefer what its own log says is
            # left of it; failing that, take off the time it has already run.
            remaining = self.active.pass_remaining
            if remaining is not None:
                total += remaining - self.mean_elapsed
            elif self.active.elapsed is not None:
                total -= min(self.active.elapsed, self.mean_elapsed)
        return max(total, 0.0)


def read_tail(path: Path, limit: int = TAIL_BYTES) -> str:
    """The last `limit` bytes of a file, decoded leniently.

    Binary-mode seek rather than `read_text`: a render log grows without bound
    on a retry loop, and the caller only wants the end of it. A multi-byte
    character cut in half by the seek is replaced, not raised.
    """
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def parse_tail(text: str) -> dict[str, Any]:
    """Pull the newest step line and heartbeat out of a log tail.

    Read backwards and stop at the first of each: the tail holds the whole
    history of the pass, and only its last line is now.
    """
    found: dict[str, Any] = {}
    for line in reversed(text.splitlines()):
        if "step" not in found:
            match = STEP_LINE.search(line)
            if match:
                rate = float(match["rate"])
                found["step"] = int(match["step"])
                found["steps"] = int(match["steps"])
                found["seconds_per_step"] = (
                    rate if match["unit"] == "s/it" else (1 / rate if rate else None)
                )
        if "elapsed" not in found:
            match = BEAT_LINE.search(line)
            if match:
                found["elapsed"] = int(match["min"] or 0) * 60 + int(match["sec"])
        if "step" in found and "elapsed" in found:
            break
    return found


def latest_log(logs_dir: Path) -> Path | None:
    """The most recently written log in a run, or None."""
    if not logs_dir.is_dir():
        return None
    logs = list(logs_dir.glob("*.log"))
    if not logs:
        return None
    try:
        return max(logs, key=lambda p: p.stat().st_mtime)
    except OSError:
        return None


def active_scene(script: Script, table: list[dict[str, Any]], *,
                 now: float | None = None) -> Active | None:
    """The scene being rendered right now, as far as disk can tell.

    Two conditions, both necessary. The newest log has to have been written
    recently — an old run leaves its logs behind forever. And the scene it
    belongs to has to still be *pending*: a scene with a clip on disk is one
    whose log is simply the last thing that run wrote, and a scene recorded as
    failed is not being retried by anyone.
    """
    log = latest_log(script.layout.logs_dir)
    if log is None:
        return None
    try:
        quiet = (now if now is not None else time.time()) - log.stat().st_mtime
    except OSError:
        return None
    if quiet > ACTIVE_SECONDS:
        return None

    name = LOG_NAME.match(log.name)
    if not name:
        return None
    row = next((r for r in table if r["slug"] == name["slug"]), None)
    if row is None or row["state"] != "pending":
        return None

    return Active(
        index=row["index"],
        id=row["id"],
        slug=row["slug"],
        log_name=log.name,
        attempt=int(name["attempt"]),
        quiet=max(0.0, quiet),
        **parse_tail(read_tail(log)),
    )


def run_progress(script: Script, table: list[dict[str, Any]], *,
                 now: float | None = None) -> RunProgress:
    """Count the scene table and add the scene in flight.

    Takes the table rather than the script alone so the page computes
    `scene_rows` — which recomputes every fingerprint — exactly once.
    """
    progress = RunProgress(total=len(table))
    for row in table:
        state = row["state"]
        if state == "rendered":
            progress.rendered += 1
        elif state == "stale":
            progress.stale += 1
        elif state == "failed":
            progress.failed += 1
        else:
            progress.pending += 1

    timings = [row["elapsed_s"] for row in table if row.get("elapsed_s")]
    if timings:
        progress.mean_elapsed = sum(timings) / len(timings)

    progress.active = active_scene(script, table, now=now)
    return progress
