"""Submitting to ComfyUI and collecting the result.

Hermetic: no server, no sockets. The pure parsers are fed canned /history
payloads, and the runners get a fake urlopen. What is being checked is that a
ComfyUI failure reaches the render loop as a non-zero exit code, so the existing
retry and backoff drive it exactly as they drive a container that exited badly.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from moviemakr.backends import comfy
from moviemakr.errors import ConfigError

PROMPT_ID = "1e1c0b7a-0000-4000-8000-000000000000"


def history(status="success", completed=True, outputs=None, messages=None):
    return {PROMPT_ID: {
        "status": {"status_str": status, "completed": completed,
                   "messages": messages or []},
        "outputs": outputs if outputs is not None else {
            "save": {"images": [
                {"filename": "001-opening_00001_.mp4",
                 "subfolder": "moviemakr/test-movie", "type": "output"},
            ]},
        },
    }}


# --- reading /history -----------------------------------------------------


def test_finished_and_successful():
    entry = comfy.history_entry(history(), PROMPT_ID)
    assert comfy.is_finished(entry)
    assert comfy.failure_reason(entry) is None


def test_missing_entry_is_not_finished():
    """Right after submitting, /history is empty rather than pending."""
    assert comfy.history_entry({}, PROMPT_ID) is None
    assert not comfy.is_finished(None)


def test_execution_error_is_reported_with_its_node():
    entry = comfy.history_entry(history(
        status="error",
        messages=[["execution_error", {
            "node_type": "MiniMaxH3AddGuide",
            "exception_message": "frame_idx 0 is past the end",
        }]],
    ), PROMPT_ID)
    reason = comfy.failure_reason(entry)
    assert "MiniMaxH3AddGuide" in reason
    assert "past the end" in reason


def test_a_non_success_status_without_messages_still_fails():
    entry = comfy.history_entry(history(status="error", messages=[]), PROMPT_ID)
    assert comfy.failure_reason(entry) == "error"


def test_savevideo_reports_under_images_not_video():
    """Reading `video` would find nothing and look like a failed render."""
    files = comfy.saved_files(comfy.history_entry(history(), PROMPT_ID))
    assert [f["filename"] for f in files] == ["001-opening_00001_.mp4"]


def test_no_outputs_is_no_files():
    assert comfy.saved_files(comfy.history_entry(history(outputs={}), PROMPT_ID)) == []


def test_output_path_joins_the_subfolder(tmp_path):
    item = {"filename": "a.mp4", "subfolder": "moviemakr/m", "type": "output"}
    assert comfy.output_path(tmp_path, item) == tmp_path / "moviemakr/m/a.mp4"


def test_output_path_without_a_subfolder(tmp_path):
    assert comfy.output_path(tmp_path, {"filename": "a.mp4"}) == tmp_path / "a.mp4"


def test_view_url_is_escaped():
    url = comfy.view_url("http://h:8188", {"filename": "a b.mp4", "subfolder": "x/y"})
    assert "filename=a+b.mp4" in url
    assert "subfolder=x%2Fy" in url
    assert "type=output" in url


# --- posting --------------------------------------------------------------


def test_payload_carries_the_ids():
    payload = comfy.prompt_payload({"a": 1}, PROMPT_ID, "client")
    assert payload == {"prompt": {"a": 1}, "prompt_id": PROMPT_ID, "client_id": "client"}


def test_a_rejected_graph_surfaces_its_node_errors(monkeypatch):
    """A 400 carries exactly which input the server disliked; keep it."""
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(json.dumps({"node_errors": {
                "unet": {"errors": [{"message": "Value not in list"}]}}}).encode()),
        )

    monkeypatch.setattr(comfy.urllib.request, "urlopen", boom)
    with pytest.raises(ConfigError) as exc:
        comfy.post_json("http://h:8188/prompt", {})
    assert "Value not in list" in str(exc.value)
    assert "unet" in str(exc.value)


# --- preflight ------------------------------------------------------------


def object_info(unet=("a.safetensors",), clip=("t.safetensors",),
                vae=("v.safetensors", "av.safetensors"), lora=("l.safetensors",)):
    def node(field, options):
        return {"input": {"required": {field: [list(options), {}]}}}
    return {
        "UNETLoader": node("unet_name", unet),
        "CLIPLoader": node("clip_name", clip),
        "VAELoader": node("vae_name", vae),
        "LoraLoaderModelOnly": node("lora_name", lora),
    }


@pytest.fixture
def comfy_models():
    return {
        "diffusion_model": "a.safetensors",
        "text_encoder": "t.safetensors",
        "video_vae": "v.safetensors",
        "audio_vae": "av.safetensors",
    }


def test_preflight_passes_when_the_server_has_the_models(
        monkeypatch, load_comfy, comfy_models):
    script = load_comfy({"comfy": comfy_models})
    monkeypatch.setattr(comfy, "get_json", lambda url, timeout=None: object_info())
    ok, message = comfy.check_server(script)
    assert ok, message


def test_preflight_names_the_model_the_server_lacks(
        monkeypatch, load_comfy, comfy_models):
    script = load_comfy({"comfy": comfy_models})
    monkeypatch.setattr(
        comfy, "get_json",
        lambda url, timeout=None: object_info(unet=("something-else.safetensors",)))
    ok, message = comfy.check_server(script)
    assert not ok
    assert "a.safetensors" in message


def test_preflight_reports_an_unreachable_server(monkeypatch, load_comfy):
    def boom(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(comfy, "get_json", boom)
    ok, message = comfy.check_server(load_comfy())
    assert not ok
    assert "cannot reach ComfyUI" in message


# --- collecting -----------------------------------------------------------


def test_collect_copies_from_a_shared_output_dir(tmp_path, load_comfy, comfy_models):
    out = tmp_path / "comfy-out"
    (out / "moviemakr/test-movie").mkdir(parents=True)
    (out / "moviemakr/test-movie/001-opening_00001_.mp4").write_bytes(b"movie-bytes")
    script = load_comfy({"comfy": {**comfy_models, "output_dir": str(out)}})

    dest = tmp_path / "scenes" / "001-opening.webm"
    assert comfy.collect(comfy.history_entry(history(), PROMPT_ID), dest, script.comfy)
    assert dest.read_bytes() == b"movie-bytes"


def test_collect_falls_back_to_http_when_the_dir_is_not_shared(
        tmp_path, monkeypatch, load_comfy):
    script = load_comfy()  # no output_dir configured

    class FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(comfy, "_request",
                        lambda url, data=None, timeout=None: FakeResp(b"downloaded"))
    dest = tmp_path / "scenes" / "001-opening.webm"
    assert comfy.collect(comfy.history_entry(history(), PROMPT_ID), dest, script.comfy)
    assert dest.read_bytes() == b"downloaded"


def test_collect_reports_failure_when_nothing_was_saved(tmp_path, load_comfy):
    script = load_comfy()
    entry = comfy.history_entry(history(outputs={}), PROMPT_ID)
    assert not comfy.collect(entry, tmp_path / "x.webm", script.comfy)


# --- the exit codes the retry loop reads ----------------------------------


def serve(status="success", outputs=None, messages=None):
    """A /history responder keyed by the id in the URL.

    run_scene mints its own prompt id, so a fake keyed by a fixed constant would
    never match and the wait loop would spin forever - which is exactly how the
    missing queue check was found.
    """
    def _get(url, timeout=None):
        if "/history/" in url:
            pid = url.rsplit("/", 1)[-1]
            return {pid: history(status=status, outputs=outputs,
                                 messages=messages)[PROMPT_ID]}
        return {"queue_running": [], "queue_pending": []}

    return _get


def run_with(monkeypatch, load_comfy, tmp_path, *, post=None, get=None, collect=True):
    (tmp_path / "out").mkdir(exist_ok=True)  # must exist before the script loads
    script = load_comfy({"comfy": {"output_dir": str(tmp_path / "out")}})
    monkeypatch.setattr(comfy, "POLL_SECONDS", 0)
    monkeypatch.setattr(comfy, "post_json", post or (lambda url, payload, timeout=None: {}))
    monkeypatch.setattr(comfy, "get_json", get or serve())
    monkeypatch.setattr(comfy, "collect", lambda entry, dest, cfg: collect)
    graph = comfy.build_graph(script.scenes[0], script)
    return comfy.run_scene(graph, script, tmp_path / "clip.webm", tmp_path / "log.txt")


def test_success_is_zero(monkeypatch, load_comfy, tmp_path):
    assert run_with(monkeypatch, load_comfy, tmp_path) == 0


def test_a_rejected_graph_is_non_zero(monkeypatch, load_comfy, tmp_path):
    def reject(url, payload, timeout=None):
        raise ConfigError("nope")

    assert run_with(monkeypatch, load_comfy, tmp_path, post=reject) == 2


def test_an_unreachable_server_is_non_zero(monkeypatch, load_comfy, tmp_path):
    def refuse(url, payload, timeout=None):
        raise urllib.error.URLError("refused")

    assert run_with(monkeypatch, load_comfy, tmp_path, post=refuse) == 3


def test_an_execution_error_is_non_zero(monkeypatch, load_comfy, tmp_path):
    assert run_with(monkeypatch, load_comfy, tmp_path, get=serve(
        status="error",
        messages=[["execution_error",
                   {"node_type": "VAEDecode", "exception_message": "boom"}]],
    )) == 1


def test_a_vanished_prompt_stops_instead_of_waiting_forever(
        monkeypatch, load_comfy, tmp_path):
    """Absent from history is normal while rendering; absent from both is not."""
    def gone(url, timeout=None):
        return {} if "/history/" in url else {"queue_running": [], "queue_pending": []}

    assert run_with(monkeypatch, load_comfy, tmp_path, get=gone) == 5


def test_a_queued_prompt_is_waited_for(monkeypatch, load_comfy, tmp_path):
    """Still in the queue and not yet in history: keep polling, do not bail."""
    seen = {"polls": 0}

    def pending_then_done(url, timeout=None):
        if "/queue" in url:
            return {"queue_running": [[0, PID["id"]]], "queue_pending": []}
        seen["polls"] += 1
        pid = url.rsplit("/", 1)[-1]
        PID["id"] = pid
        if seen["polls"] < 3:
            return {}
        return {pid: history()[PROMPT_ID]}

    PID = {"id": ""}
    assert run_with(monkeypatch, load_comfy, tmp_path, get=pending_then_done) == 0
    assert seen["polls"] == 3


def test_an_uncollectable_output_is_non_zero(monkeypatch, load_comfy, tmp_path):
    """Succeeding without producing a file must not count as a rendered scene."""
    assert run_with(monkeypatch, load_comfy, tmp_path, collect=False) == 4


def test_the_log_records_the_graph(monkeypatch, load_comfy, tmp_path):
    run_with(monkeypatch, load_comfy, tmp_path)
    log = (tmp_path / "log.txt").read_text()
    assert "MiniMaxH3ImageToVideo" in log
    assert "/prompt" in log


def test_a_transient_poll_error_does_not_fail_the_render(monkeypatch, load_comfy, tmp_path):
    """The server going quiet while busy is not a failed render."""
    calls = {"n": 0}

    responder = serve()

    def flaky(url, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("busy")
        return responder(url)

    assert run_with(monkeypatch, load_comfy, tmp_path, get=flaky) == 0
    assert calls["n"] == 2
