"""Summary formatting. Pure: builds lines, prints nothing but the final join."""

from __future__ import annotations

from pathlib import Path
from typing import Any

WIDTH = 68


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(int(seconds), 60)
    if m:
        return f"{m}m{s:02d}s"
    return f"{seconds:.1f}s"


def fmt_span(seconds: float | None) -> str:
    """A rough duration, to the minute: "2h19m", "19m", "45s".

    Separate from `fmt_duration` because it is for estimates, not measurements -
    a scene took 25m44s, but the hours a render still has to go should not
    pretend to that kind of precision, and "139m01s" does not read as anything.
    """
    if seconds is None:
        return "-"
    minutes, secs = divmod(int(seconds), 60)
    if not minutes:
        return f"{secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def format_summary(
    results: list[dict[str, Any]],
    logs_dir: Path,
    movie: Path | None,
) -> list[str]:
    """Render the per-scene table, totals, failure footer and movie line.

    `results` rows are {"scene", "state", "probe", "elapsed"}.
    """
    lines = ["", "=" * WIDTH]
    lines.append(f"{'#':>3}  {'scene':<22} {'state':<9} {'length':>8} {'elapsed':>9}")
    lines.append("-" * WIDTH)

    total_len = 0.0
    total_elapsed = 0.0
    for row in results:
        scene = row["scene"]
        probe = row.get("probe") or {}
        duration = probe.get("duration")
        if duration:
            total_len += duration
        if row.get("elapsed"):
            total_elapsed += row["elapsed"]
        lines.append(
            f"{scene.index:>3}  {scene.id[:22]:<22} {row['state']:<9} "
            f"{fmt_duration(duration):>8} {fmt_duration(row.get('elapsed')):>9}"
        )

    lines.append("-" * WIDTH)
    lines.append(
        f"{'':>3}  {'total':<22} {'':<9} {fmt_duration(total_len):>8} "
        f"{fmt_duration(total_elapsed):>9}"
    )

    failures = [r for r in results if r["state"] == "failed"]
    if failures:
        ids = ", ".join(r["scene"].id for r in failures)
        lines.append(f"\n{len(failures)} scene(s) failed: {ids}")
        lines.append(f"logs: {logs_dir}")

    stale = [r for r in results if r["state"] == "stale"]
    if stale:
        ids = ", ".join(r["scene"].id for r in stale)
        lines.append(f"\n{len(stale)} scene(s) stale (script changed since render): {ids}")

    if movie and movie.is_file():
        size_mb = movie.stat().st_size / (1024 * 1024)
        lines.append(f"\nmovie: {movie}  ({size_mb:.1f} MB)")

    return lines


def print_summary(results: list[dict[str, Any]], logs_dir: Path, movie: Path | None) -> None:
    print("\n".join(format_summary(results, logs_dir, movie)))
