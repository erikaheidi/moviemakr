"""ffmpeg / ffprobe work.

Deliberately depends on nothing but `errors`: the command builders are pure and
the runners take plain paths, so this module is testable without a Script and
without invoking ffmpeg.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

# Codec pair per output container; used for normalization and the final mux.
CONTAINERS = {
    "mp4": {"vcodec": "libx264", "acodec": "aac", "vargs": ["-crf", "18", "-preset", "medium"]},
    "webm": {"vcodec": "libvpx-vp9", "acodec": "libopus", "vargs": ["-crf", "30", "-b:v", "0"]},
}

# Reference videos are always expanded at 24fps, whatever the scene's own fps.
REFVIDEO_FPS = 24
REFVIDEO_FRAME_WARN = 240


@dataclass(frozen=True, slots=True)
class NormalizeSpec:
    width: int
    height: int
    fps: int
    container: str
    keep_audio: bool

    @property
    def codecs(self) -> dict:
        return CONTAINERS[self.container]


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


def ffprobe_json(path: Path) -> dict:
    """Never raises; an unreadable file is reported as {}."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def probe_clip(path: Path) -> dict:
    """Real frame count and duration - the model does not honour --video-frames exactly."""
    info = ffprobe_json(path)
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration = None
    fmt_dur = (info.get("format") or {}).get("duration")
    if fmt_dur:
        try:
            duration = float(fmt_dur)
        except ValueError:
            duration = None

    frames = None
    if video is not None:
        nb = video.get("nb_frames")
        if nb and nb.isdigit():
            frames = int(nb)
        else:
            # WebM often omits nb_frames; counting is slower but exact.
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True,
            )
            out = proc.stdout.strip()
            if out.isdigit():
                frames = int(out)

    return {
        "frames": frames,
        "duration": duration,
        "has_audio": has_audio,
        "width": (video or {}).get("width"),
        "height": (video or {}).get("height"),
    }


def extract_last_frame(clip: Path, dest: Path) -> bool:
    """Grab the final video frame. -sseof is cheap; reverse is the fallback."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempts = [
        ["ffmpeg", "-v", "error", "-y", "-sseof", "-0.5", "-i", str(clip),
         "-update", "1", "-frames:v", "1", str(dest)],
        ["ffmpeg", "-v", "error", "-y", "-i", str(clip),
         "-vf", "reverse", "-frames:v", "1", str(dest)],
    ]
    for cmd in attempts:
        if dest.exists():
            dest.unlink()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
            return True
    return False


# --------------------------------------------------------------------------
# tail clips - the overlap anchor handed to the next scene
# --------------------------------------------------------------------------


def tail_start(duration: float, frames: int, fps: int) -> float:
    """Seek offset that leaves `frames` frames at the end of a clip."""
    return max(0.0, duration - frames / fps)


def tail_clip_cmd(src: Path, dest: Path, *, start: float, frames: int) -> list[str]:
    """Cut the last `frames` frames of a clip, audio included, into one file.

    Video and audio travel together on purpose: the next scene anchors both, and
    a separate frame dump plus a separate wav would have to be re-aligned by
    hand. Re-encoded rather than stream-copied because `-c copy` seeks to a
    keyframe, which would silently deliver a different number of frames than the
    assembly trim later removes.

    Near-lossless (crf 12): this clip is a conditioning anchor, and its artefacts
    would be baked into the next scene.
    """
    return [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{start:.6f}", "-i", str(src),
        "-frames:v", str(frames),
        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(dest),
    ]


def extract_tail_clip(clip: Path, dest: Path, frames: int, fps: int) -> bool:
    """Write the last `frames` frames of `clip` (with audio) to `dest`."""
    if frames < 1:
        return False
    duration = probe_clip(clip).get("duration")
    if not duration:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    cmd = tail_clip_cmd(clip, dest, start=tail_start(duration, frames, fps), frames=frames)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0


# --------------------------------------------------------------------------
# stills - evenly spaced frames pulled out of a rendered clip
# --------------------------------------------------------------------------


def still_timestamps(duration: float, count: int) -> list[float]:
    """Midpoints of `count` equal segments, not endpoints.

    Midpoints, because 0.0 and `duration` land on the first and last frames -
    which on a turnaround are the two most motion-blurred and least useful
    views, and the last one is already covered by `extract_last_frame`.
    """
    if count < 1:
        raise ConfigError(f"need at least one still, got {count}")
    if duration <= 0:
        raise ConfigError(f"clip has no measurable duration: {duration}")
    return [duration * (i + 0.5) / count for i in range(count)]


def still_cmd(clip: Path, dest: Path, at: float) -> list[str]:
    """One frame at `at` seconds. -ss before -i is the fast input seek."""
    return [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{at:.3f}",
        "-i", str(clip),
        "-frames:v", "1", "-update", "1",
        str(dest),
    ]


def extract_still(clip: Path, dest: Path, at: float) -> bool:
    """Write one frame; True when a non-empty file landed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    proc = subprocess.run(still_cmd(clip, dest, at), capture_output=True, text=True)
    return proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0


# --------------------------------------------------------------------------
# reference video -> frame directory
# --------------------------------------------------------------------------


def source_stamp(src: Path, width: int, height: int) -> str:
    """Identity of an extracted frame directory.

    Written to `.source` and folded into the scene fingerprint, so the format is
    load-bearing: changing it invalidates every chain that uses a ref video.
    """
    stat = src.stat()
    return f"{src.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{width}x{height}"


def refvideo_filter(width: int, height: int) -> str:
    """Centre-crop to the target aspect, then scale - never squash.

    A landscape source reframed into a portrait movie keeps the subject's
    proportions instead of distorting it.
    """
    return (
        f"fps={REFVIDEO_FPS},"
        f"crop='min(iw,ih*{width}/{height})':'min(ih,iw*{height}/{width})',"
        f"scale={width}:{height}"
    )


def prepare_ref_video(src: Path, frame_dir: Path, width: int, height: int) -> Path:
    """Turn a reference video into the 24fps frame directory the model expects.

    Re-extracts only when the source file or the target resolution changes.
    """
    stamp = frame_dir / ".source"
    signature = source_stamp(src, width, height)

    if stamp.is_file() and stamp.read_text() == signature and any(frame_dir.glob("*.png")):
        return frame_dir

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-vf", refvideo_filter(width, height),
         "-start_number", "0", str(frame_dir / "%04d.png")],
        check=True, capture_output=True, text=True,
    )

    count = len(list(frame_dir.glob("*.png")))
    if count == 0:
        raise ConfigError(f"reference video produced no frames: {src}")
    if count > REFVIDEO_FRAME_WARN:
        print(f"  note: {src.name} expands to {count} reference frames "
              f"({count / REFVIDEO_FPS:.0f}s); long references cost memory", file=sys.stderr)

    stamp.write_text(signature)
    return frame_dir


# --------------------------------------------------------------------------
# assembly command builders (pure)
# --------------------------------------------------------------------------


def normalize_cmd(src: Path, dest: Path, spec: NormalizeSpec, *, has_audio: bool,
                  skip_seconds: float = 0.0) -> list[str]:
    """Re-encode one clip to uniform codecs/size/fps so `concat -c copy` is safe.

    The generator writes PCM audio into WebM, which is off-spec and does not
    stream-copy reliably; clips are also free to differ in size. Scale-and-pad,
    never stretch.

    `skip_seconds` drops the head of the clip, which is how an overlap-chained
    scene stops repeating the frames it was anchored on. The seek goes *before*
    `-i`, so video and audio are trimmed as one and stay in sync; trimming with
    a filter would move only the video and slide the soundtrack forward.
    """
    codecs = spec.codecs
    vf = (
        f"scale={spec.width}:{spec.height}:force_original_aspect_ratio=decrease,"
        f"pad={spec.width}:{spec.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={spec.fps},format=yuv420p"
    )

    cmd = ["ffmpeg", "-v", "error", "-y"]
    if skip_seconds > 0:
        cmd += ["-ss", f"{skip_seconds:.6f}"]
    cmd += ["-i", str(src)]
    needs_silence = spec.keep_audio and not has_audio
    if needs_silence:
        # Give every output the same stream layout, even for a silent clip.
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]

    cmd += ["-map", "0:v:0"]
    if spec.keep_audio:
        cmd += ["-map", "1:a:0" if needs_silence else "0:a:0"]

    cmd += ["-vf", vf, "-c:v", codecs["vcodec"], *codecs["vargs"], "-r", str(spec.fps)]
    if spec.keep_audio:
        cmd += ["-c:a", codecs["acodec"], "-ar", "48000", "-ac", "2", "-b:a", "192k"]
        if needs_silence:
            cmd += ["-shortest"]
    else:
        cmd += ["-an"]

    return cmd + [str(dest)]


def concat_list_text(paths: list[Path]) -> str:
    """ffmpeg concat demuxer list. Single quotes inside a path close and re-open."""
    return "".join(f"file '{p.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
                   for p in paths)


def concat_cmd(list_file: Path, dest: Path) -> list[str]:
    return ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", str(dest)]


def music_mix_cmd(src: Path, music: Path, dest: Path, gain_db: float, acodec: str) -> list[str]:
    """Loop the bed under the whole movie; `duration=first` stops at the video."""
    return [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(src),
        "-stream_loop", "-1", "-i", str(music),
        "-filter_complex",
        f"[1:a]volume={gain_db}dB,aresample=48000[bed];"
        f"[0:a][bed]amix=inputs=2:duration=first:dropout_transition=0[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", acodec, "-b:a", "192k",
        str(dest),
    ]


def normalize_clip(src: Path, dest: Path, spec: NormalizeSpec, *,
                   skip_seconds: float = 0.0) -> None:
    probe = probe_clip(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(normalize_cmd(src, dest, spec, has_audio=bool(probe.get("has_audio")),
                             skip_seconds=skip_seconds))


def run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)
