"""Unknown YAML keys are a hard error with a suggestion.

They used to be dropped in silence, so `video_frame: 90` did nothing and you
found out after the render finished.
"""

from __future__ import annotations

import pytest

from moviemakr.errors import ConfigError, check_keys, suggest


def test_scene_typo_is_caught_with_a_suggestion(load):
    with pytest.raises(ConfigError) as exc:
        load({"scenes": [{"id": "a", "prompt": "p", "video_frame": 90}]})
    message = str(exc.value)
    assert "video_frame" in message
    assert "did you mean 'video_frames'?" in message
    assert "scene 1 ('a')" in message


def test_chain_typo_is_caught(load):
    with pytest.raises(ConfigError) as exc:
        load({"scenes": [{"id": "a", "prompt": "p", "chain_from_previou": True}]})
    assert "did you mean 'chain_from_previous'?" in str(exc.value)


@pytest.mark.parametrize(
    "overrides,where",
    [
        ({"nonsense": 1}, "top level"),
        ({"model": {"difusion_model": "x"}}, "model"),
        ({"docker": {"imagee": "x"}}, "docker"),
        ({"defaults": {"wdith": 512}}, "defaults"),
        ({"continuity": {"anchor": []}}, "continuity"),
        ({"output": {"containr": "mp4"}}, "output"),
    ],
)
def test_every_block_rejects_unknown_keys(load, overrides, where):
    with pytest.raises(ConfigError) as exc:
        load(overrides)
    assert where in str(exc.value)


def test_all_unknown_keys_appear_in_one_message(load):
    with pytest.raises(ConfigError) as exc:
        load({"scenes": [{"id": "a", "prompt": "p", "wdith": 1, "hieght": 2}]})
    message = str(exc.value)
    assert "wdith" in message
    assert "hieght" in message


def test_message_lists_allowed_keys(load):
    with pytest.raises(ConfigError) as exc:
        load({"output": {"zzzzzzzz": 1}})
    assert "allowed:" in str(exc.value)
    assert "container" in str(exc.value)


def test_no_bogus_suggestion_when_nothing_is_close(load):
    with pytest.raises(ConfigError) as exc:
        load({"output": {"zzzzzzzz": 1}})
    assert "did you mean" not in str(exc.value)


# --------------------------------------------------------------------------
# the helpers themselves
# --------------------------------------------------------------------------


def test_check_keys_passes_when_all_known():
    check_keys("x", {"a": 1, "b": 2}, {"a", "b", "c"})


def test_check_keys_accepts_empty():
    check_keys("x", {}, {"a"})


def test_suggest_is_deterministic():
    """Options are sorted first, so ties do not depend on set ordering."""
    options = {"seed", "steps", "speed", "sped"}
    assert suggest("sed", options) == suggest("sed", list(reversed(sorted(options))))


def test_suggest_returns_none_when_far():
    assert suggest("qqqqqqqq", {"width", "height"}) is None
