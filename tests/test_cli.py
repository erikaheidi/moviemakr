"""Argument parsing, option plumbing, and top-level error handling."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pytest

from moviemakr.cli import build_parser, check_tools, main
from moviemakr.render import RenderOptions


@pytest.fixture
def tools_present(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.mark.parametrize("command", ["render", "assemble", "status"])
def test_subcommands_parse(command):
    args = build_parser().parse_args([command, "script.yaml"])
    assert args.command == command
    assert args.script == Path("script.yaml")


def test_render_defaults():
    args = build_parser().parse_args(["render", "s.yaml"])
    assert args.retries == 2
    assert args.force is False
    assert args.dry_run is False
    assert args.only is None and args.scene is None


def test_only_and_scene_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["render", "s.yaml", "--only", "1", "--scene", "a"])


def test_no_command_is_an_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_render_options_round_trip_every_flag():
    args = build_parser().parse_args([
        "render", "s.yaml", "--only", "2,4-6", "--force", "--retries", "5",
        "--halt-on-failure", "--no-assemble", "--dry-run", "--allow-cpu",
    ])
    opts = RenderOptions.from_args(args)
    assert opts == RenderOptions(
        only="2,4-6", scene=None, force=True, retries=5,
        halt_on_failure=True, no_assemble=True, dry_run=True, allow_cpu=True,
    )


def test_attempts_is_retries_plus_one():
    assert RenderOptions(retries=2).attempts == 3
    assert RenderOptions(retries=0).attempts == 1
    assert RenderOptions(retries=-5).attempts == 1


# --------------------------------------------------------------------------
# tool checks
# --------------------------------------------------------------------------


def test_missing_ffmpeg_is_exit_2(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert check_tools("status", False) == 2
    assert "ffmpeg not found" in capsys.readouterr().err


def test_docker_only_required_for_a_real_render(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "docker" else "/usr/bin/x")
    assert check_tools("render", False) == 2
    assert check_tools("render", True) is None  # --dry-run needs no docker
    assert check_tools("status", False) is None
    assert check_tools("assemble", False) is None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def test_missing_script_is_exit_2(tools_present, capsys, tmp_path, project_root):
    code = main(["status", str(tmp_path / "nope.yaml")], project_root=project_root)
    assert code == 2
    assert "error: script not found" in capsys.readouterr().err


def test_config_error_is_exit_2(tools_present, capsys, write_script, project_root):
    path = write_script({"scenes": [{"id": "a", "prompt": "p", "video_frame": 90}]})
    code = main(["status", str(path)], project_root=project_root)
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "did you mean 'video_frames'?" in err


def test_status_runs(tools_present, capsys, write_script, project_root):
    path = write_script()
    assert main(["status", str(path)], project_root=project_root) == 0
    out = capsys.readouterr().out
    assert "opening" in out
    assert "pending" in out  # nothing rendered yet


def test_assemble_without_clips_is_exit_1(tools_present, capsys, write_script, project_root):
    path = write_script()
    assert main(["assemble", str(path)], project_root=project_root) == 1
    assert "no rendered clips found" in capsys.readouterr().err


def test_called_process_error_names_the_program(tools_present, capsys, write_script,
                                                project_root, monkeypatch):
    """It used to print 'ffmpeg -v error failed', which says nothing."""
    from moviemakr import cli

    def boom(script):
        raise subprocess.CalledProcessError(
            1, ["ffmpeg", "-v", "error", "-i", "x.webm"], stderr="Invalid data found\n"
        )

    monkeypatch.setattr(cli, "cmd_assemble", boom)
    path = write_script()
    assert main(["assemble", str(path)], project_root=project_root) == 1
    err = capsys.readouterr().err
    assert "ffmpeg failed (exit 1)" in err
    assert "Invalid data found" in err
    assert "command: ffmpeg -v error -i x.webm" in err


def test_keyboard_interrupt_is_exit_130(tools_present, capsys, write_script,
                                        project_root, monkeypatch):
    from moviemakr import cli

    def boom(script):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_status", boom)
    path = write_script()
    assert main(["status", str(path)], project_root=project_root) == 130
    assert "interrupted" in capsys.readouterr().err


def test_workspace_flag_beats_the_default(tools_present, capsys, write_script,
                                          project_root, tmp_path):
    """--workspace overrides the fallback root, so the run dir moves with it."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "assets").mkdir(parents=True)
    path = write_script()

    assert main(["status", "--workspace", str(elsewhere), str(path)],
                project_root=project_root) == 0
    # The status table prints the run dir's logs path; the movie lives under it.
    assert not (project_root / "renders").exists()


def test_workspace_must_exist(tools_present, capsys, write_script, project_root, tmp_path):
    path = write_script()
    code = main(["status", "--workspace", str(tmp_path / "nope"), str(path)],
                project_root=project_root)
    assert code == 2
    assert "workspace is not a directory" in capsys.readouterr().err
