"""The docker wrapper around sd-cli.

The GPU group-add is the load-bearing part: without those groups Vulkan
enumerates nothing and ggml silently falls back to CPU, at hours per scene.
"""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest

from moviemakr.backends.sdcpp import (
    check_gpu,
    container_name,
    device_gids,
    docker_argv,
    docker_base_argv,
    format_argv,
)


@pytest.fixture
def fake_gids(monkeypatch):
    """Report deterministic gids for device nodes that do not exist on this box.

    Everything outside /dev is delegated to the real os.stat - pathlib uses it
    internally, so a blanket fake would break ordinary file operations.
    """
    gids = {"/dev/dri/card1": 44, "/dev/dri/renderD128": 991, "/dev/kfd": 991}
    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        name = str(path)
        if name in gids:
            return SimpleNamespace(st_gid=gids[name])
        if name.startswith("/dev/"):
            raise OSError(f"no such device: {name}")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fake_stat)
    return gids


DEVICES = ["/dev/dri/card1", "/dev/dri/renderD128", "/dev/kfd"]


def test_device_gids_dedupes_in_order(fake_gids):
    assert device_gids(DEVICES) == [44, 991]


def test_device_gids_skips_unstattable_nodes(fake_gids):
    assert device_gids(["/dev/nope", "/dev/dri/card1"]) == [44]


def test_device_gids_of_nothing():
    assert device_gids([]) == []


def test_base_argv_order(load, fake_gids):
    script = load({"docker": {"devices": DEVICES}})
    argv = docker_base_argv(script)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[3:5] == ["--user", f"{os.getuid()}:{os.getgid()}"]
    assert argv[5:9] == ["--group-add", "44", "--group-add", "991"]
    assert argv[9:] == [
        "--device", "/dev/dri/card1",
        "--device", "/dev/dri/renderD128",
        "--device", "/dev/kfd",
    ]


def test_run_as_root_drops_user_and_groups_but_keeps_devices(load, fake_gids):
    script = load({"docker": {"devices": DEVICES, "run_as_current_user": False}})
    argv = docker_base_argv(script)
    assert "--user" not in argv
    assert "--group-add" not in argv
    assert argv.count("--device") == 3


def test_container_name(load):
    script = load({"name": "Cats Cooking"})
    assert container_name(script.scenes[0], script) == "moviemakr-cats-cooking-001-opening"


def test_full_argv_shape(load, fake_gids):
    script = load({"docker": {"devices": []}})
    scene = script.scenes[0]
    argv = docker_argv(scene, script, [])
    layout = script.layout

    assert argv[argv.index("--name") + 1] == container_name(scene, script)
    assert "-t" in argv  # TTY, so sd-cli line-buffers its progress

    # All three mounts sit immediately before the image, in order, with models
    # and assets read-only and the run dir writable.
    image_at = argv.index(script.docker.image)
    assert argv[image_at - 6:image_at] == [
        "-v", f"{layout.model_root}:/models:ro",
        "-v", f"{layout.assets_dir}:/assets:ro",
        "-v", f"{layout.run_dir}:/out",
    ][:6]
    assert argv[image_at - 2:image_at] == ["-v", f"{layout.run_dir}:/out"]

    # sd args come after the image.
    assert argv[image_at + 1] == "--mode"


# --------------------------------------------------------------------------
# format_argv
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["docker", "run"], "docker run"),
        (["--prompt", "a b"], "--prompt 'a b'"),
        (["--backend", "te=cpu"], "--backend te=cpu"),
        (["-W", "540"], "-W 540"),
    ],
)
def test_format_argv(argv, expected):
    assert format_argv(argv) == expected


def test_format_argv_quotes_apostrophes_shell_safely():
    quoted = format_argv(["--prompt", "the world's sauce"])
    # However it escapes, the shell must read it back as one original token.
    read_back = subprocess.run(
        ["sh", "-c", f"printf %s {quoted.split(' ', 1)[1]}"],
        capture_output=True, text=True, check=True,
    )
    assert read_back.stdout == "the world's sauce"


# --------------------------------------------------------------------------
# GPU preflight
# --------------------------------------------------------------------------


def fake_run(monkeypatch, *, stdout="", stderr="", returncode=0):
    def _run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", _run)


def test_gpu_detected(load, monkeypatch):
    fake_run(monkeypatch, stdout="Vulkan0\tAMD Radeon\nCPU\tfallback\n")
    ok, message = check_gpu(load())
    assert ok
    assert "Vulkan0" in message


def test_cpu_only_is_rejected(load, monkeypatch):
    fake_run(monkeypatch, stdout="CPU\tfallback\n")
    ok, message = check_gpu(load())
    assert not ok
    assert "fall back to CPU" in message


def test_no_devices_is_rejected(load, monkeypatch):
    fake_run(monkeypatch, stdout="No devices found\n")
    ok, _ = check_gpu(load())
    assert not ok


def test_load_backend_lines_are_ignored(load, monkeypatch):
    fake_run(monkeypatch, stdout="load_backend\tvulkan.so\nCPU\tfallback\n")
    ok, _ = check_gpu(load())
    assert not ok


def test_container_failure_reports_the_real_error(load, monkeypatch):
    """It used to blame the GPU when the container never started at all."""
    fake_run(monkeypatch, returncode=125, stderr="docker: invalid reference format\n")
    ok, message = check_gpu(load())
    assert not ok
    assert "invalid reference format" in message
    assert "exited 125" in message
    assert "no GPU backend visible" not in message


def test_timeout_continues_rather_than_blocking_the_run(load, monkeypatch):
    def _raise(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 300)

    monkeypatch.setattr(subprocess, "run", _raise)
    ok, message = check_gpu(load())
    assert ok
    assert "could not verify" in message
