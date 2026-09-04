"""Summary formatting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from moviemakr.report import fmt_duration, fmt_span, format_summary

LOGS = Path("/out/logs")


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (None, "-"),
        (0, "0.0s"),
        (9.44, "9.4s"),
        (59.9, "59.9s"),
        (60, "1m00s"),
        (61, "1m01s"),
        (3661, "61m01s"),
    ],
)
def test_fmt_duration(seconds, expected):
    assert fmt_duration(seconds) == expected


def row(index=1, scene_id="opening", state="rendered", duration=5.0, elapsed=60.0):
    return {
        "scene": SimpleNamespace(index=index, id=scene_id),
        "state": state,
        "probe": {"duration": duration},
        "elapsed": elapsed,
    }


def test_header_and_row():
    lines = format_summary([row()], LOGS, None)
    assert any("scene" in line and "elapsed" in line for line in lines)
    assert any("opening" in line and "rendered" in line for line in lines)


def test_totals_add_up():
    lines = format_summary([row(elapsed=60.0), row(2, "b", elapsed=120.0)], LOGS, None)
    total = next(line for line in lines if "total" in line)
    assert "10.0s" in total  # 5 + 5
    assert "3m00s" in total  # 60 + 120


def test_totals_skip_missing_values():
    lines = format_summary([row(duration=None, elapsed=None)], LOGS, None)
    total = next(line for line in lines if "total" in line)
    assert "0.0s" in total


def test_long_scene_id_is_truncated():
    lines = format_summary([row(scene_id="a" * 40)], LOGS, None)
    assert "a" * 22 in "\n".join(lines)
    assert "a" * 23 not in "\n".join(lines)


def test_failure_footer_lists_ids_and_logs():
    lines = format_summary(
        [row(state="failed"), row(2, "b", state="failed")], LOGS, None
    )
    text = "\n".join(lines)
    assert "2 scene(s) failed: opening, b" in text
    assert "logs: /out/logs" in text


def test_no_failure_footer_when_all_good():
    assert "failed" not in "\n".join(format_summary([row()], LOGS, None))


def test_stale_footer():
    text = "\n".join(format_summary([row(state="stale")], LOGS, None))
    assert "1 scene(s) stale" in text
    assert "opening" in text


def test_movie_line_shows_size(tmp_path):
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"x" * (2 * 1024 * 1024))
    text = "\n".join(format_summary([row()], LOGS, movie))
    assert "movie:" in text
    assert "2.0 MB" in text


def test_no_movie_line_when_absent(tmp_path):
    text = "\n".join(format_summary([row()], LOGS, tmp_path / "gone.mp4"))
    assert "movie:" not in text


def test_empty_results_does_not_crash():
    lines = format_summary([], LOGS, None)
    assert any("total" in line for line in lines)


# --- estimates -------------------------------------------------------------


def test_fmt_span_rolls_over_into_hours():
    """The web view's estimate: "139m01s" does not read as two hours."""
    assert fmt_span(8341) == "2h19m"
    assert fmt_span(1142) == "19m"
    assert fmt_span(45) == "45s"
    assert fmt_span(3600) == "1h00m"
    assert fmt_span(None) == "-"
