"""ffmpeg command construction. Nothing is executed here."""

from __future__ import annotations

from pathlib import Path

import pytest

from moviemakr.errors import ConfigError
from moviemakr.media import (
    NormalizeSpec,
    concat_cmd,
    concat_list_text,
    music_mix_cmd,
    normalize_cmd,
    refvideo_filter,
    source_stamp,
    still_cmd,
    still_timestamps,
)

SRC = Path("/out/scenes/001-a.webm")
DEST = Path("/out/normalized/001-a.mp4")


def spec(container="mp4", keep_audio=True, width=540, height=960, fps=24):
    return NormalizeSpec(width=width, height=height, fps=fps,
                         container=container, keep_audio=keep_audio)


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------


def test_video_filter_scales_and_pads_never_stretches():
    cmd = normalize_cmd(SRC, DEST, spec(), has_audio=True)
    assert cmd[cmd.index("-vf") + 1] == (
        "scale=540:960:force_original_aspect_ratio=decrease,"
        "pad=540:960:(ow-iw)/2:(oh-ih)/2:color=black,"
        "fps=24,format=yuv420p"
    )


def test_keep_audio_with_audio_maps_the_source_stream():
    cmd = normalize_cmd(SRC, DEST, spec(), has_audio=True)
    assert "anullsrc" not in " ".join(cmd)
    assert "-shortest" not in cmd
    assert cmd[cmd.index("-map") + 1] == "0:v:0"
    assert "0:a:0" in cmd


def test_keep_audio_without_audio_adds_silence():
    """Every normalized clip needs the same stream layout, or concat cannot copy."""
    cmd = normalize_cmd(SRC, DEST, spec(), has_audio=False)
    joined = " ".join(cmd)
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in joined
    assert "1:a:0" in cmd
    assert "-shortest" in cmd


def test_strip_audio():
    cmd = normalize_cmd(SRC, DEST, spec(keep_audio=False), has_audio=True)
    assert "-an" in cmd
    assert "-c:a" not in cmd
    assert "anullsrc" not in " ".join(cmd)


def test_mp4_codecs():
    cmd = normalize_cmd(SRC, DEST, spec("mp4"), has_audio=True)
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "18"
    assert "-preset" in cmd and cmd[cmd.index("-preset") + 1] == "medium"


def test_webm_codecs():
    cmd = normalize_cmd(SRC, DEST, spec("webm"), has_audio=True)
    assert cmd[cmd.index("-c:v") + 1] == "libvpx-vp9"
    assert cmd[cmd.index("-c:a") + 1] == "libopus"
    assert cmd[cmd.index("-crf") + 1] == "30"
    assert cmd[cmd.index("-b:v") + 1] == "0"


def test_audio_is_normalized_to_stereo_48k():
    cmd = normalize_cmd(SRC, DEST, spec(), has_audio=True)
    assert cmd[cmd.index("-ar") + 1] == "48000"
    assert cmd[cmd.index("-ac") + 1] == "2"


def test_output_frame_rate_is_forced():
    cmd = normalize_cmd(SRC, DEST, spec(fps=30), has_audio=True)
    assert cmd[cmd.index("-r") + 1] == "30"


def test_src_and_dest_positions():
    cmd = normalize_cmd(SRC, DEST, spec(), has_audio=True)
    assert cmd[:5] == ["ffmpeg", "-v", "error", "-y", "-i"]
    assert cmd[5] == str(SRC)
    assert cmd[-1] == str(DEST)


# --------------------------------------------------------------------------
# concat + music
# --------------------------------------------------------------------------


def test_concat_list_text():
    text = concat_list_text([Path("/out/normalized/a.mp4"), Path("/out/normalized/b.mp4")])
    assert text == "file '/out/normalized/a.mp4'\nfile '/out/normalized/b.mp4'\n"


def test_concat_list_escapes_single_quotes():
    text = concat_list_text([Path("/out/it's here.mp4")])
    assert text == "file '/out/it'\\''s here.mp4'\n"


def test_concat_list_of_nothing():
    assert concat_list_text([]) == ""


def test_concat_is_a_stream_copy():
    cmd = concat_cmd(Path("/out/concat.txt"), Path("/out/movie.mp4"))
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "concat"
    assert cmd[cmd.index("-safe") + 1] == "0"
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert cmd[-1] == "/out/movie.mp4"


def test_music_mix_keeps_video_and_stops_at_the_video():
    cmd = music_mix_cmd(Path("/out/tmp.mp4"), Path("/assets/bed.mp3"),
                        Path("/out/movie.mp4"), -18, "aac")
    joined = " ".join(cmd)
    assert "volume=-18dB" in joined
    assert "duration=first" in joined
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[-1] == "/out/movie.mp4"


# --------------------------------------------------------------------------
# reference video
# --------------------------------------------------------------------------


def test_refvideo_filter_crops_then_scales_at_24fps():
    assert refvideo_filter(544, 960) == (
        "fps=24,"
        "crop='min(iw,ih*544/960)':'min(ih,iw*960/544)',"
        "scale=544:960"
    )


def test_refvideo_fps_is_fixed_regardless_of_scene_fps():
    assert refvideo_filter(100, 100).startswith("fps=24,")


def test_source_stamp_format(tmp_path):
    """This string is folded into the fingerprint, so its shape is load-bearing."""
    src = tmp_path / "clip.webm"
    src.write_bytes(b"1234567890")
    stamp = source_stamp(src, 544, 960)
    parts = stamp.split("|")
    assert parts[0] == str(src.resolve())
    assert parts[1] == "10"
    assert parts[2] == str(int(src.stat().st_mtime))
    assert parts[3] == "544x960"


def test_source_stamp_changes_with_resolution(tmp_path):
    src = tmp_path / "clip.webm"
    src.write_bytes(b"x")
    assert source_stamp(src, 540, 960) != source_stamp(src, 544, 960)


# --------------------------------------------------------------------------
# stills
# --------------------------------------------------------------------------


def test_still_timestamps_are_segment_midpoints():
    assert still_timestamps(4.0, 4) == [0.5, 1.5, 2.5, 3.5]


def test_still_timestamps_never_hit_either_end():
    """0.0 and the duration are the two blurriest frames of a turnaround."""
    stamps = still_timestamps(3.75, 6)
    assert len(stamps) == 6
    assert stamps[0] > 0
    assert stamps[-1] < 3.75
    assert stamps == sorted(stamps)


def test_one_still_is_the_middle_of_the_clip():
    assert still_timestamps(10.0, 1) == [5.0]


@pytest.mark.parametrize("duration,count", [(4.0, 0), (4.0, -1), (0.0, 6), (-2.0, 6)])
def test_still_timestamps_rejects_bad_input(duration, count):
    with pytest.raises(ConfigError):
        still_timestamps(duration, count)


def test_still_cmd_seeks_before_the_input():
    """-ss ahead of -i is the fast seek; after it, ffmpeg decodes the whole clip."""
    cmd = still_cmd(SRC, Path("/assets/cat-01.png"), 1.25)
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "1.250"
    assert cmd[cmd.index("-i") + 1] == str(SRC)


def test_still_cmd_asks_for_exactly_one_frame():
    cmd = still_cmd(SRC, Path("/assets/cat-01.png"), 0.5)
    assert cmd[cmd.index("-frames:v") + 1] == "1"
    assert "-update" in cmd
    assert cmd[-1] == "/assets/cat-01.png"
