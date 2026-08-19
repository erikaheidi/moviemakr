"""SceneSettings: typed, frozen, and merged from defaults + per-scene overrides."""

from __future__ import annotations

import dataclasses

import pytest

from moviemakr.config import SceneSettings
from moviemakr.errors import ConfigError

# These are the pre-refactor DEFAULTS. They feed the fingerprint, so a change
# here silently re-renders every scene of every existing run.
LEGACY_DEFAULTS = {
    "width": 540,
    "height": 960,
    "fps": 24,
    "video_frames": 120,
    "cfg_scale": 1.0,
    "seed": 42,
    "steps": None,
    "negative_prompt": "",
    "style_suffix": "",
    "sampling_method": None,
    "extra_args": (),
}


def test_defaults_match_legacy():
    settings = SceneSettings()
    for key, expected in LEGACY_DEFAULTS.items():
        assert getattr(settings, key) == expected, key


def test_fields_covers_every_dataclass_field():
    declared = {f.name for f in dataclasses.fields(SceneSettings)}
    assert SceneSettings.FIELDS == declared


def test_merge_overrides_only_given_keys():
    merged = SceneSettings().merge({"seed": 7}, "defaults")
    assert merged.seed == 7
    assert merged.width == 540


def test_merge_layers_scene_over_script_defaults():
    script_defaults = SceneSettings().merge({"seed": 1, "video_frames": 90}, "defaults")
    scene = script_defaults.merge({"seed": 2}, "scene 1")
    assert (scene.seed, scene.video_frames) == (2, 90)


def test_merge_returns_a_new_instance():
    base = SceneSettings()
    assert base.merge({"seed": 9}, "x") is not base
    assert base.seed == 42


def test_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SceneSettings().seed = 7


def test_slots_reject_a_typo_attribute():
    """The same class of bug that strict YAML keys eliminates, at runtime.

    frozen+slots rejects an unknown attribute with TypeError rather than
    AttributeError (the frozen __setattr__ reaches super() before slots can
    complain). What matters is that the assignment does not silently succeed.
    """
    settings = SceneSettings()
    with pytest.raises((AttributeError, TypeError)):
        settings.video_frame = 90
    assert not hasattr(settings, "video_frame")


def test_extra_args_is_an_immutable_tuple():
    settings = SceneSettings().merge({"extra_args": ["--mmap"]}, "defaults")
    assert settings.extra_args == ("--mmap",)
    assert isinstance(settings.extra_args, tuple)


def test_extra_args_default_is_not_shared():
    assert SceneSettings().extra_args is SceneSettings().extra_args  # the empty tuple
    a = SceneSettings().merge({"extra_args": ["--x"]}, "d")
    assert SceneSettings().extra_args == ()
    assert a.extra_args == ("--x",)


def test_explicit_null_is_the_same_as_omitting():
    assert SceneSettings().merge({"steps": None}, "d").steps is None
    assert SceneSettings().merge({"sampling_method": None}, "d").sampling_method is None


# --------------------------------------------------------------------------
# coercion
# --------------------------------------------------------------------------


def test_numeric_strings_are_coerced():
    merged = SceneSettings().merge({"width": "544", "cfg_scale": "2.5"}, "d")
    assert merged.width == 544
    assert merged.cfg_scale == 2.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": "wide"},
        {"height": None},
        {"fps": True},
        {"video_frames": [1]},
        {"cfg_scale": "loud"},
        {"seed": "abc"},
        {"steps": "many"},
        {"extra_args": "--mmap"},  # a bare string is a common mistake
        {"negative_prompt": {"a": 1}},
    ],
)
def test_bad_types_raise(overrides):
    with pytest.raises(ConfigError):
        SceneSettings().merge(overrides, "scene 1 ('x')")


def test_error_names_the_key_and_the_scene():
    with pytest.raises(ConfigError) as exc:
        SceneSettings().merge({"width": "wide"}, "scene 3 ('kitchen')")
    message = str(exc.value)
    assert "scene 3 ('kitchen')" in message
    assert "width" in message


@pytest.mark.parametrize("key", ["width", "height", "fps", "video_frames"])
def test_dimensions_must_be_positive(key):
    with pytest.raises(ConfigError):
        SceneSettings().merge({key: 0}, "defaults")


def test_seed_may_be_negative():
    assert SceneSettings().merge({"seed": -1}, "d").seed == -1
