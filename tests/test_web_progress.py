"""Progress inferred from a render that nothing here is running.

No FastAPI needed: `progress` takes a Script plus the scene table `browse`
already built, and returns plain dataclasses. The logs are written by hand -
the two lines each backend prints are the whole contract this module has with
them, so they are what the tests pin.
"""

from __future__ import annotations

import os
import time

import pytest

from moviemakr.web import browse as B
from moviemakr.web import progress as P

# One line of each backend's live output, copied from a real run.
SDCPP_TAIL = """\
[INFO ] stable-diffusion.cpp:1234 - sampling using Euler method
  |=========>                                        | 3/20 - 175.60s/it\x1b[K
  |=============>                                    | 4/20 - 175.59s/it\x1b[K
"""
COMFY_TAIL = """\
POST http://localhost:8188/prompt  prompt_id=abc

  [2/10] still rendering (12m32s)
  [2/10] still rendering (13m02s)
"""


@pytest.fixture
def rendering(web_workspace):
    """The `simple` script mid-render: scene two has a log and no clip.

    `web_workspace` leaves scene one rendered with a stored elapsed, so this is
    the ordinary case - one scene done, the next one in flight.
    """
    script = B.load_script_at(web_workspace, "simple.yaml")
    log = script.layout.logs_dir / "002-middle.attempt1.log"

    def _start(text: str = SDCPP_TAIL, *, age: float = 0.0, name: str | None = None):
        path = script.layout.logs_dir / name if name else log
        path.write_text(text)
        if age:
            stamp = time.time() - age
            os.utime(path, (stamp, stamp))
        return script

    return _start


def table(script):
    return B.scene_table(script)


# --- parsing ---------------------------------------------------------------


def test_reads_the_sdcpp_sampler_line():
    found = P.parse_tail(SDCPP_TAIL)
    assert (found["step"], found["steps"]) == (4, 20)
    assert found["seconds_per_step"] == pytest.approx(175.59)


def test_it_per_second_is_inverted_into_seconds_per_step():
    """sd-cli flips the unit once a pass runs faster than a second per step."""
    found = P.parse_tail("  |=====>    | 5/15 - 2.50it/s\n")
    assert found["seconds_per_step"] == pytest.approx(0.4)


def test_reads_the_comfy_heartbeat():
    assert P.parse_tail(COMFY_TAIL)["elapsed"] == 13 * 60 + 2


def test_a_heartbeat_under_a_minute_has_no_minutes_part():
    assert P.parse_tail("  [1/3] still rendering (45s)\n")["elapsed"] == 45


def test_the_last_line_wins():
    """The tail holds the whole pass; only its final line is now."""
    assert P.parse_tail(SDCPP_TAIL)["step"] == 4


def test_a_log_with_nothing_to_say_parses_to_nothing():
    assert P.parse_tail("loading diffusion model\nusing mmap\n") == {}


def test_read_tail_returns_only_the_end(tmp_path):
    path = tmp_path / "big.log"
    path.write_text("x" * 9000 + "TAIL\n")
    text = P.read_tail(path, limit=64)
    assert text.endswith("TAIL\n")
    assert len(text) <= 64


def test_read_tail_of_a_missing_file_is_empty(tmp_path):
    assert P.read_tail(tmp_path / "nope.log") == ""


# --- what is running -------------------------------------------------------


def test_a_fresh_log_for_a_pending_scene_is_the_active_scene(rendering):
    script = rendering()
    active = P.active_scene(script, table(script))
    assert active is not None
    assert (active.id, active.slug) == ("middle", "002-middle")
    assert active.attempt == 1
    assert (active.step, active.steps) == (4, 20)
    assert active.pass_percent == 20
    assert active.pass_remaining == pytest.approx(16 * 175.59)


def test_an_old_log_is_not_a_running_render(rendering):
    """Logs outlive the run that wrote them; only a recent write means live."""
    script = rendering(age=P.ACTIVE_SECONDS + 60)
    assert P.active_scene(script, table(script)) is None


def test_a_scene_that_already_has_a_clip_is_not_rendering(rendering):
    """The last log of a finished run stays fresh for minutes after it ends."""
    script = rendering(name="001-opening.attempt2.log")
    assert P.active_scene(script, table(script)) is None


def test_a_scene_recorded_as_failed_is_not_rendering(rendering, web_workspace):
    import json

    script = rendering()
    state = json.loads(script.layout.state_file.read_text())
    state["scenes"]["middle"] = {"fingerprint": None, "state": "failed"}
    script.layout.state_file.write_text(json.dumps(state))
    assert P.active_scene(script, table(script)) is None


def test_a_retry_reports_its_attempt_number(rendering):
    script = rendering(name="002-middle.attempt3.log")
    active = P.active_scene(script, table(script))
    assert active is not None and active.attempt == 3


def test_no_logs_dir_is_not_an_error(web_workspace):
    script = B.load_script_at(web_workspace, "h3/beach.yaml")
    assert not script.layout.logs_dir.exists()
    assert P.active_scene(script, table(script)) is None


def test_a_log_that_is_not_a_scene_log_is_ignored(rendering):
    """`logs/` is served over HTTP; something else could land in it."""
    script = rendering(name="notes.log")
    assert P.active_scene(script, table(script)) is None


# --- the run as a whole ----------------------------------------------------


def test_counts_and_percentage(web_workspace):
    script = B.load_script_at(web_workspace, "simple.yaml")
    run = P.run_progress(script, table(script))
    assert (run.total, run.rendered, run.pending) == (2, 1, 1)
    assert run.percent == 50
    assert run.outstanding == 1


def test_nothing_rendered_is_zero_not_a_division_error(web_workspace):
    script = B.load_script_at(web_workspace, "h3/beach.yaml")
    run = P.run_progress(script, table(script))
    assert run.percent == 0
    assert run.eta is None          # nothing has finished, so no pace to go on
    assert not run.running


def test_the_estimate_uses_the_mean_of_the_scenes_that_finished(rendering):
    """One 902.5s scene done, one in flight: the estimate is what its log says."""
    script = rendering()
    run = P.run_progress(script, table(script))
    assert run.mean_elapsed == pytest.approx(902.5)
    assert run.eta == pytest.approx(16 * 175.59)


def test_a_comfy_run_estimates_from_the_heartbeat(rendering):
    script = rendering(COMFY_TAIL)
    run = P.run_progress(script, table(script))
    assert run.active is not None and run.active.elapsed == 782
    assert run.eta == pytest.approx(902.5 - 782)


def test_an_estimate_never_goes_negative(rendering):
    """A scene can run longer than the mean; the answer is 0, not -20m."""
    script = rendering("  [2/2] still rendering (60m00s)\n")
    assert P.run_progress(script, table(script)).eta == 0


def test_a_finished_run_has_no_estimate(web_workspace):
    """Everything rendered: there is nothing left to be an estimate of."""
    script = B.load_script_at(web_workspace, "simple.yaml")
    rows = table(script)
    for row in rows:
        row["state"] = "rendered"
    run = P.run_progress(script, rows)
    assert run.outstanding == 0
    assert run.eta is None


def test_the_sliver_is_one_scene_wide(web_workspace):
    script = B.load_script_at(web_workspace, "simple.yaml")
    assert P.run_progress(script, table(script)).slice_percent == 50
