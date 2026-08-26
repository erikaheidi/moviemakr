"""The launcher and the package share a name; make sure the package wins."""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_resolves_to_the_package_not_the_launcher():
    """`moviemakr.py` and `moviemakr/` sit side by side; the directory wins."""
    import moviemakr

    assert Path(moviemakr.__file__).name == "__init__.py"
    assert Path(moviemakr.__file__).parent.name == "moviemakr"


def test_launcher_compiles():
    py_compile.compile(str(REPO_ROOT / "moviemakr.py"), doraise=True)


def test_public_api_is_importable():
    import moviemakr

    for name in moviemakr.__all__:
        assert hasattr(moviemakr, name), name


def test_python_m_moviemakr_help():
    proc = subprocess.run(
        [sys.executable, "-m", "moviemakr", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "render" in proc.stdout


def test_launcher_runs_and_reports_a_bad_script(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "moviemakr.py"), "status",
         "--workspace", str(tmp_path), "definitely-not-here.yaml"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "script not found" in proc.stderr


def test_launcher_without_a_workspace_says_so():
    """The checkout is not a fallback workspace, so this must not half-work."""
    env = {k: v for k, v in os.environ.items() if k != "MOVIEMAKR_WORKSPACE"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "moviemakr.py"), "status", "anything.yaml"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 2
    assert "no workspace" in proc.stderr


def test_no_import_cycles():
    """Every module must import cleanly on its own, in any order."""
    modules = [
        "moviemakr.errors", "moviemakr.layout", "moviemakr.report", "moviemakr.state",
        "moviemakr.media", "moviemakr.config", "moviemakr.docker", "moviemakr.assemble",
        "moviemakr.render", "moviemakr.cli",
    ]
    for module in modules:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"{module}: {proc.stderr}"
