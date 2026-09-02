"""The backend registry and the script's `backend:` key.

A wrong backend name must fail at load time. Silently falling back to the default
engine would render a whole script with the wrong one, at hours apiece.
"""

from __future__ import annotations

import pytest

from moviemakr.backends import BACKENDS, DEFAULT_BACKEND, check_name, resolve
from moviemakr.errors import ConfigError


def test_default_backend_is_registered():
    assert DEFAULT_BACKEND in BACKENDS


def test_check_name_accepts_registered():
    assert check_name("sdcpp") == "sdcpp"


def test_check_name_rejects_unknown():
    with pytest.raises(ConfigError) as exc:
        check_name("nope")
    assert "unknown backend" in str(exc.value)
    assert "sdcpp" in str(exc.value)


def test_check_name_suggests_a_near_miss():
    with pytest.raises(ConfigError) as exc:
        check_name("sdcp")
    assert "did you mean 'sdcpp'" in str(exc.value)


def test_resolve_returns_the_module():
    module = resolve("sdcpp")
    assert hasattr(module, "sd_args")
    assert hasattr(module, "fingerprint")


def test_resolve_rejects_unknown():
    with pytest.raises(ConfigError):
        resolve("nope")


# --- the script key -------------------------------------------------------


def test_script_defaults_to_sdcpp(load):
    assert load().backend == "sdcpp"


def test_script_accepts_an_explicit_backend(load):
    assert load({"backend": "sdcpp"}).backend == "sdcpp"


def test_script_rejects_an_unknown_backend(load):
    with pytest.raises(ConfigError) as exc:
        load({"backend": "comfyui"})
    assert "unknown backend" in str(exc.value)


def test_backend_is_a_known_top_level_key(load):
    """It must not trip the unknown-key check that guards every other typo."""
    script = load({"backend": "sdcpp"})
    assert script.backend == "sdcpp"
