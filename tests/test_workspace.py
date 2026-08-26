"""Workspace: where the data root comes from, and what hangs off it.

The point of this type is that the data (scripts, assets, drafts, renders) no
longer has to live inside the code checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moviemakr.errors import ConfigError
from moviemakr.layout import WORKSPACE_ENV, Workspace


@pytest.fixture
def a_dir(tmp_path: Path):
    def _make(name: str) -> Path:
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    return _make


# --- resolution order ------------------------------------------------------


def test_explicit_wins_over_env(a_dir, monkeypatch):
    explicit, env = a_dir("explicit"), a_dir("env")
    monkeypatch.setenv(WORKSPACE_ENV, str(env))
    assert Workspace.resolve(explicit).root == explicit


def test_env_is_used_without_an_explicit_argument(a_dir, monkeypatch):
    env = a_dir("env")
    monkeypatch.setenv(WORKSPACE_ENV, str(env))
    assert Workspace.resolve(None).root == env


def test_no_workspace_at_all_is_an_error():
    """There is no third fallback - the checkout is not a workspace."""
    with pytest.raises(ConfigError, match=WORKSPACE_ENV):
        Workspace.resolve(None)


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="workspace is not a directory"):
        Workspace.resolve(tmp_path / "nope")


def test_empty_env_var_counts_as_unset(monkeypatch):
    """An exported-but-empty var means "unset", not "the current directory"."""
    monkeypatch.setenv(WORKSPACE_ENV, "")
    with pytest.raises(ConfigError, match=WORKSPACE_ENV):
        Workspace.resolve(None)


# --- derived paths ---------------------------------------------------------


def test_subdirectories(tmp_path):
    ws = Workspace.at(tmp_path)
    assert ws.scripts_dir == tmp_path / "scripts"
    assert ws.assets_dir == tmp_path / "assets"
    assert ws.drafts_dir == tmp_path / "drafts"
    assert ws.renders_dir == tmp_path / "renders"
    assert ws.cache_dir == tmp_path / ".cache"


def test_root_is_resolved_and_user_expanded(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert Workspace.at(tmp_path / "a" / ".." / "a" / "b").root == nested.resolve()


def test_home_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert Workspace.at(Path("~/work")).root == (tmp_path / "work").resolve()


# --- the reason it exists --------------------------------------------------


def test_script_loads_against_a_workspace_with_no_checkout(load, workspace, project_root):
    """assets/ and renders/ resolve inside the workspace, wherever that is."""
    script = load()
    assert script.workspace == workspace
    assert script.assets_dir == (project_root / "assets").resolve()
    assert script.run_dir == (project_root / "renders" / "test-movie").resolve()
    assert not (project_root / "moviemakr").exists()  # no code here
