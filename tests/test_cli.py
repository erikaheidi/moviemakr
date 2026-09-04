"""Argument parsing, option plumbing, and top-level error handling."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from moviemakr.cli import build_parser, check_tools, main
from moviemakr.errors import ConfigError
from moviemakr.render import RenderOptions


@pytest.fixture
def tools_present(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.mark.parametrize("command", ["render", "assemble", "status", "stills"])
def test_subcommands_parse(command):
    args = build_parser().parse_args([command, "script.yaml"])
    assert args.command == command
    assert args.script == Path("script.yaml")


def test_stills_defaults():
    args = build_parser().parse_args(["stills", "s.yaml"])
    assert args.count == 6
    assert args.scene is None
    assert args.dest is None
    assert args.prefix is None


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
    code = main(["status", str(tmp_path / "nope.yaml"), "--workspace", str(project_root)])
    assert code == 2
    assert "error: script not found" in capsys.readouterr().err


def test_config_error_is_exit_2(tools_present, capsys, write_script, project_root):
    path = write_script({"scenes": [{"id": "a", "prompt": "p", "video_frame": 90}]})
    code = main(["status", str(path), "--workspace", str(project_root)])
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "did you mean 'video_frames'?" in err


def test_status_runs(tools_present, capsys, write_script, project_root):
    path = write_script()
    assert main(["status", str(path), "--workspace", str(project_root)]) == 0
    out = capsys.readouterr().out
    assert "opening" in out
    assert "pending" in out  # nothing rendered yet


def test_assemble_without_clips_is_exit_1(tools_present, capsys, write_script, project_root):
    path = write_script()
    assert main(["assemble", str(path), "--workspace", str(project_root)]) == 1
    assert "no rendered clips found" in capsys.readouterr().err


# --------------------------------------------------------------------------
# stills
# --------------------------------------------------------------------------


def test_pick_scene_defaults_to_the_only_scene(load):
    from moviemakr.cli import pick_scene

    script = load()
    assert pick_scene(script, None).id == "opening"


def test_pick_scene_needs_an_id_when_there_are_several(load):
    from moviemakr.cli import pick_scene

    script = load({"scenes": [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "p"}]})
    with pytest.raises(ConfigError) as exc:
        pick_scene(script, None)
    assert "a, b" in str(exc.value)


def test_pick_scene_lists_the_ids_on_a_bad_one(load):
    from moviemakr.cli import pick_scene

    script = load({"scenes": [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "p"}]})
    with pytest.raises(ConfigError) as exc:
        pick_scene(script, "nope")
    assert "available: a, b" in str(exc.value)


def test_stills_without_a_clip_is_exit_1(tools_present, capsys, write_script, project_root):
    path = write_script()
    assert main(["stills", str(path), "--workspace", str(project_root)]) == 1
    assert "no rendered clip" in capsys.readouterr().err


def test_stills_with_a_bad_scene_id_is_exit_2(tools_present, capsys, write_script,
                                              project_root):
    path = write_script()
    code = main(["stills", str(path), "--scene", "nope", "--workspace", str(project_root)])
    assert code == 2
    assert "available: opening" in capsys.readouterr().err


def test_stills_writes_into_assets_by_default(tools_present, capsys, load, write_script,
                                              project_root, monkeypatch):
    """The default destination is assets/, the only mount a ref image can live under."""
    from moviemakr import cli

    path = write_script()
    script = load()
    clip = script.layout.clip(script.scenes[0].slug)
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"not-really-a-webm")

    monkeypatch.setattr(cli, "probe_clip", lambda p: {"duration": 4.0})
    written = []

    def fake_extract(clip, dest, at):
        written.append((dest, at))
        return True

    monkeypatch.setattr(cli, "extract_still", fake_extract)

    assert main(["stills", str(path), "--count", "2", "--workspace", str(project_root)]) == 0
    # The slug carries the scene index, so the default prefix does too.
    assert [d.name for d, _ in written] == ["001-opening-01.png", "001-opening-02.png"]
    assert all(d.parent == (project_root / "assets").resolve() for d, _ in written)
    assert [at for _, at in written] == [1.0, 3.0]


def test_stills_reports_a_failed_extraction(tools_present, capsys, load, write_script,
                                            project_root, monkeypatch):
    from moviemakr import cli

    path = write_script()
    script = load()
    clip = script.layout.clip(script.scenes[0].slug)
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x")

    monkeypatch.setattr(cli, "probe_clip", lambda p: {"duration": 4.0})
    monkeypatch.setattr(cli, "extract_still", lambda c, d, a: False)

    assert main(["stills", str(path), "--count", "1", "--workspace", str(project_root)]) == 1
    assert "FAILED" in capsys.readouterr().err


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
    assert main(["assemble", str(path), "--workspace", str(project_root)]) == 1
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
    assert main(["status", str(path), "--workspace", str(project_root)]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_workspace_flag_beats_the_env_var(tools_present, capsys, write_script,
                                          project_root, tmp_path, monkeypatch):
    """--workspace overrides $MOVIEMAKR_WORKSPACE, so the run dir moves with it."""
    from moviemakr.layout import WORKSPACE_ENV

    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "assets").mkdir(parents=True)
    monkeypatch.setenv(WORKSPACE_ENV, str(project_root))
    path = write_script()

    assert main(["status", "--workspace", str(elsewhere), str(path)]) == 0
    # The status table prints the run dir's logs path; the movie lives under it.
    assert not (project_root / "renders").exists()


def test_workspace_must_exist(tools_present, capsys, write_script, project_root, tmp_path):
    path = write_script()
    code = main(["status", "--workspace", str(tmp_path / "nope"), str(path)])
    assert code == 2
    assert "workspace is not a directory" in capsys.readouterr().err


def test_no_workspace_at_all_is_exit_2(tools_present, capsys, write_script):
    """There is no checkout fallback: the workspace has to be named."""
    path = write_script()
    assert main(["status", str(path)]) == 2
    assert "no workspace" in capsys.readouterr().err


# --- serve -----------------------------------------------------------------


def test_serve_reports_a_missing_web_extra_instead_of_crashing(
    tools_present, monkeypatch, capsys, tmp_path
):
    """The hint is only reachable if the check does not rely on ImportError.

    `moviemakr.web` imports nothing but the stdlib, so `from .web import
    run_server` always succeeds - uvicorn is imported inside `run_server`, and
    the ModuleNotFoundError used to escape past the guard as a traceback.
    """
    import moviemakr.web as W

    monkeypatch.setattr(W, "missing_modules", lambda: ["uvicorn"])
    code = main(["serve", "--workspace", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "web extra is not installed" in err
    assert "missing uvicorn" in err
    assert "pip install 'moviemakr[web]'" in err


def test_serve_starts_when_the_extra_is_present(tools_present, monkeypatch, tmp_path):
    import moviemakr.web as W

    seen = {}
    monkeypatch.setattr(W, "missing_modules", lambda: [])
    monkeypatch.setattr(W, "run_server", lambda ws, **kw: seen.update(ws=ws, **kw) or 0)
    assert main(["serve", "--workspace", str(tmp_path), "--port", "9001"]) == 0
    assert seen["port"] == 9001
    assert seen["host"] == "127.0.0.1"
    assert seen["reload"] is False
    assert seen["ws"].root == tmp_path.resolve()


def test_missing_modules_is_empty_when_the_extra_is_installed():
    """Sanity check on the probe itself, in the venv that has the extra."""
    pytest.importorskip("fastapi", reason="install the 'web' extra to run this")
    from moviemakr.web import missing_modules

    assert missing_modules() == []
