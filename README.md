# moviemakr

Renders a multi-scene movie from a YAML script with MiniMax-H3, one scene at a
time, then stitches the clips into a single movie with ffmpeg. Two engines can do
the rendering — `stable-diffusion.cpp` in Docker, or a ComfyUI server — chosen per
script with a single key.

The model can only produce a few seconds of video per run. `moviemakr` turns that
constraint into a workflow: you write the whole movie as an ordered list of scene
prompts, and it renders them in sequence, resumes where it left off, and assembles
the result.

## The workspace

Your content — `scripts/`, `assets/`, `drafts/`, `renders/` — lives in a
**workspace** directory, separate from this checkout, so the tool can be
installed anywhere and several workspaces can coexist:

```
~/moviemakr-workspace/
  scripts/     the YAML, nested however you like
  assets/      reference images
  drafts/      plain-prose notes, before they become scripts
  renders/     output; disposable, reproducible from the scripts
```

Point at it with `--workspace PATH` or `$MOVIEMAKR_WORKSPACE`. One of the two is
required — this checkout holds code only, and is not a fallback workspace.

## Usage

```bash
export MOVIEMAKR_WORKSPACE=~/moviemakr-workspace
cd "$MOVIEMAKR_WORKSPACE"

moviemakr render   scripts/beach.yaml --dry-run   # print commands, render nothing
moviemakr render   scripts/beach.yaml             # render everything, then assemble
moviemakr status   scripts/beach.yaml             # per-scene state and real durations
moviemakr assemble scripts/beach.yaml             # re-stitch from existing clips
moviemakr serve                                   # browse it all in a browser
```

Useful flags on `render`:

| flag | effect |
| --- | --- |
| `--only 2,4-6` | render a subset by scene index |
| `--scene kitchen,feast` | render a subset by scene id |
| `--force` | re-render even when up to date |
| `--retries N` | attempts per scene beyond the first (default 2) |
| `--halt-on-failure` | stop the run at the first failing scene |
| `--no-assemble` | render clips but skip the final stitch |
| `--allow-cpu` | skip the GPU preflight and render anyway (`sdcpp` only) |

Start with `--dry-run`. It prints exactly what each scene would submit — the
`docker run` command on `sdcpp`, the API graph on `comfy` — including
reference-image wiring, without spending GPU time. It writes nothing, not even
the extracted frames for a reference video, and needs neither Docker nor a
running ComfyUI.

`--only` and `--scene` both validate their argument, so a mistyped id or an index
past the end of the script is an error rather than a run that quietly does
nothing.

`status` recomputes each scene's fingerprint, so it distinguishes a scene that is
genuinely up to date from one marked **stale** because the script changed since it
was rendered:

```
  #  scene                  state       length   elapsed
--------------------------------------------------------------------
  1  cooking                rendered      3.7s    42m06s
  2  serving                stale         3.7s    44m25s
```

## Backends

Two engines can render a scene, chosen with the `backend:` key. `sdcpp` is the
default, so a script that does not mention it keeps working unchanged.

| | `sdcpp` (default) | `comfy` |
| --- | --- | --- |
| how a scene runs | one `docker run` of `sd-cli` | an API graph POSTed to a running ComfyUI |
| needs | Docker, and the weights on disk | a ComfyUI already serving the models |
| model files | host paths, under `model:` | ComfyUI-side *names*, under `comfy:` |
| weights | reloaded every scene (~39 GB) | stay resident between scenes |
| chaining hands over | the previous scene's last frame | a segment of the previous clip, audio included |
| references | images and video | images only, at most 9 |
| raw clip | `.webm` | `.mp4` |
| measured, 544x960x90 | ~50 min per scene | ~18 min per scene at 14 steps |

The trade is setup against speed. `sdcpp` needs nothing running and each scene is
one self-contained command, but `sd-cli` is one-shot, so roughly 39 GB of weights
reload every time. `comfy` needs a server you have already configured, and in
exchange keeps the models loaded and can anchor a whole *segment* of the previous
scene rather than a single frame.

### Rendering through ComfyUI

`model:` and `docker:` are rejected for `backend: comfy`. The model names move
under `comfy:`, where they are ComfyUI-side names — whatever its loader dropdowns
list — rather than host paths:

```yaml
backend: comfy

comfy:
  url: http://127.0.0.1:8188          # default
  input_dir:  /srv/comfyui/input      # host side of ComfyUI's own input/
  output_dir: /srv/comfyui/output     # ...and its output/
  diffusion_model: minimax_h3_ref2va_pruned_bf16.safetensors
  text_encoder:    qwen3vl_32b_minimax_h3_bf16.safetensors
  video_vae:       minimax_h3_video_vae_fp16.safetensors
  audio_vae:       minimax_h3_audio_vae_fp32.safetensors
  steps: 14                           # default 8
  ref_image_size: max                 # or "match" (default)
```

Reference images and the anchor clip are copied into `input_dir`; the finished
mp4 is collected from `output_dir`. A local server shares those directories, so
clips are copied from disk rather than pulled back through HTTP. Both are checked
to exist when the script loads.

Because the model names are the server's, they cannot be checked against the
filesystem the way `sdcpp`'s paths are. The preflight asks the running server
instead — whether it offers each named model, and whether the graph matches its
schema — before committing to a long render.

Two settings are worth knowing about:

- **`ref_image_size`** — `match` scales each reference to the generation's pixel
  area; `max` uses a 2048px short edge. Reference tokens ride through every
  sampling step, so `max` costs real time, but `match` can scale a face down far
  enough to lose the identity the reference was added for.
- **`steps`** — not cheap at full size. Measured on gfx1151, a sampling step
  costs 8.8s at 640x384x56 and 74s at 544x960x90.

H3 also ships as two task variants, `ref2va` (reads reference images) and `fl2va`
(reads first/last keyframes). Crossing them does not error — the graph runs and
the model produces something, just not what the script describes — so a dry run
guesses from the filename and warns when the checkpoint looks wrong for how the
scenes are conditioned.

### What the two do not share

- **`cfg_scale`, `negative_prompt`, `sampling_method` and `extra_args` are inert
  on `comfy`.** H3 has no negative conditioning at CFG 1, so the graph drives it
  through `BasicGuider`; the rest are `sd-cli`'s. These keys stay *valid* — they
  are not typos, so the unknown-key check cannot catch them — and a dry run warns
  about them instead.
- **`ref_videos` and `continuity.anchor_videos` are not supported on `comfy`
  yet**, and are rejected when the script loads.
- **A scene carries at most 9 reference images on `comfy`**, the limit of the
  node's autogrow inputs.
- **Fingerprints do not transfer.** Each backend hashes what its own engine will
  produce, so switching a script's backend re-renders every scene rather than
  resuming from clips the other one made. That is deliberate: the same scene
  through the two engines is not the same output.
- **The GPU preflight differs, and only `sdcpp`'s can be skipped.** `sdcpp` asks
  the container which compute backends it can see and aborts if there is no
  accelerator, because its CPU fallback is silent and costs hours; `--allow-cpu`
  overrides that. `comfy` has no equivalent override — its check is a fast HTTP
  call against a server that is either there or not.

## Writing a script

See [`examples/example.yaml`](examples/example.yaml) for a fully commented template.
The shape is: an engine block — `model` plus `docker` for `sdcpp`, or `comfy` for
ComfyUI — then `defaults` that apply to every scene, `continuity` and `output`
policy, and an ordered `scenes` list. Everything outside the engine block is the
same on both backends.

Unknown keys are rejected when the script loads, with a suggestion — so a typo
costs you a second rather than a render:

```
error: scene 3 ('too-spicy'): unknown key(s)
  video_frame   (did you mean 'video_frames'?)
  allowed: cfg_scale, chain_from_previous, extra_args, fps, height, id, ...
```

Any key under `defaults` can be overridden on an individual scene:

```yaml
defaults:
  video_frames: 120
  seed: 42
  style_suffix: "Front camera view, indoor, cinematic."

scenes:
  - id: opening
    prompt: "Three cats dancing a choreographed routine."
    seed: 1234          # override just for this scene
  - id: kitchen
    prompt: "The cats walk into a bright kitchen."
    chain_from_previous: true
    video_frames: 86
```

`style_suffix` is appended to every prompt, so the look stays consistent without
repeating yourself in each scene.

## Keeping scenes consistent

Two independent mechanisms:

- **`continuity.anchors`** — reference images under `assets/` passed to *every*
  scene. Use this for a character sheet, so the cast looks the same throughout.
- **`chain_from_previous: true`** — hands the end of each scene to the next one,
  which gives shot-to-shot flow. Set it globally under `continuity` or per scene.

They compose: a scene can use anchors, the chained hand-off, and its own
`ref_images` at once. Scenes outside a filtered selection still contribute their
hand-off, so `--only`/`--scene` runs do not break downstream continuity.

### What chaining hands over depends on the backend

This is the one place the two engines produce visibly different results:

- On **`sdcpp`** it is a *still*. Each scene's final frame is extracted to
  `renders/<name>/frames/` and passed to the next scene as another reference.
- On **`comfy`** it is a *segment*. The tail of the previous clip — video **and
  its soundtrack** — is anchored at frame 0 of the next scene, then trimmed back
  off at assembly.

A single handed-off frame restarts both the motion and the soundscape at every
seam; a segment carries them across. That is the main reason to reach for the
comfy backend, and `sd-cli` has no way to express it.

`overlap_frames` sets how long that segment is, and it is not free: the anchor's
keyframe latents ride through *every* sampling step, exactly like reference
tokens. Measured on one scene at 544x960x90:

| overlap | cost per sampling step |
| --- | --- |
| none | 84.2s |
| 5 frames | 93.8s (+11%) |
| 22 frames | 137.8s (+64%) |

The default is **5** — the shortest length on the model's frame grid, still long
enough to carry its slice of audio, and the seam held up. A 22-frame anchor
roughly doubled the scene for 0.9s of shared motion against 0.2s.
`overlap_frames: 0` is a hard cut. The key is ignored on `sdcpp`.

The count actually anchored is written to `state.json` and assembly trims from
*that*, never from the script's current `overlap_frames` — editing the value must
not change how yesterday's clip gets cut.

### Referring to images in a prompt: `<Picture N>` shifts by backend

H3 prompts cite their references by index, and the index depends on the engine,
because the chained content takes slot 0 on `sdcpp` and no slot at all on `comfy`:

| | the first anchor is |
| --- | --- |
| `sdcpp`, scene that chains | `<Picture 2>` — the chained frame took `<Picture 1>` |
| `sdcpp`, scene that does not | `<Picture 1>` |
| `comfy` | `<Picture 1>`, chained or not |

A prompt ported between the two unchanged therefore misnumbers every reference it
cites. Nothing errors; the model simply conditions on the wrong image.

### Using a video as the reference (`sdcpp` only)

This model is Ref2VA, so it takes video references natively. A clip carries both
appearance and motion, which a single still cannot. The comfy backend does not
support this yet — `ref_videos` and `continuity.anchor_videos` are rejected there
when the script loads.

```yaml
continuity:
  anchor_videos:                                  # applies to every scene
    - /home/erika/Projects/ai-models/cat-cooking.webm

scenes:
  - id: opening
    prompt: "..."
    ref_videos: [some-other-clip.webm]            # or just this scene
```

Any video file works — moviemakr expands it into the 24 fps frame directory the
model expects, under `renders/<name>/refvideos/`, and passes that as
`--ref-video`. Frames are **centre-cropped** to the target aspect and then
scaled, never squashed, so a landscape reference used in a portrait movie keeps
the subject's proportions. Look in that directory to check the reframing before
committing to a long render.

Source paths may be relative to `assets/` or absolute, and unlike `ref_images`
they do not need to sit inside a mounted directory — the extracted frames land
under the run dir, which is mounted, so the source itself never has to be.
Frames are re-extracted only when the source file or the target resolution
changes.

A `ref_image` *does* have to be reachable from inside the container, and that is
checked when the script loads rather than part-way through the render. That
constraint is the container's, not the model's: on `comfy` any readable host path
works, because the backend copies references into ComfyUI's own input directory.

## Resume and retries

Renders are expensive, so a scene is only re-run when it actually needs to be.
Each scene is fingerprinted over its resolved command line *and the content of its
reference images*. A scene is skipped when its clip exists and the fingerprint
matches, so:

- Editing one prompt re-renders that scene only.
- If a chained scene's upstream clip changes, the chained frame's content changes
  too, and the downstream scene correctly re-renders as well.
- Ctrl-C discards the partial clip; the next run picks up from that scene.

On `sdcpp`, each scene runs in a container named `moviemakr-<script>-<scene>`, so
`docker ps` tells you which scene is rendering. Interrupting stops that container
explicitly — `docker run` is only a client, and killing it would otherwise leave
the job running under the daemon, burning CPU and RAM with nothing watching it.
If you ever suspect a stray render:

```bash
docker ps --filter name=moviemakr        # what is actually running
docker rm -f <name>                      # stop one
```

`comfy` has the same hazard in a different shape: an HTTP POST is not a client
whose death the server notices, so a killed moviemakr would leave ComfyUI
sampling a prompt nobody will collect, holding the GPU for the rest of the run.
Interrupting therefore posts `/interrupt`, and SIGTERM is bridged to the same
path as Ctrl-C so that a `kill` behaves like one.

Failures retry (default 2 extra attempts) and then move on, so one bad scene does
not cost you the batch. Every attempt is logged to `renders/<name>/logs/`, and the
run exits non-zero if anything failed. Use `--halt-on-failure` to stop instead.

## Output layout

```
<workspace>/renders/<script-name>/
  scenes/     001-opening.webm        raw model output (always WebM)
  frames/     001-opening.last.png    final frame, used for chaining
  normalized/ 001-opening.mp4         uniform codecs/size/fps for concat
  refvideos/  cat-cooking-544x960/    extracted reference-video frames
  logs/       001-opening.attempt1.log
  concat.txt                          the ffmpeg concat list
  state.json                          fingerprints, timings, probe results
  <script-name>.mp4                   the finished movie
```

`scenes/` holds whatever the engine wrote — `.webm` from `sd-cli`, `.mp4` from
ComfyUI's `SaveVideo` — and only `normalized/` and the finished movie follow
`output.container`.

`frames/` and `refvideos/` belong to `sdcpp`. A comfy run chains on the clip
itself and does not support reference videos, so it produces neither; its anchor
clips and reference images live in ComfyUI's own `input_dir` instead.

## Browsing it from a phone (`moviemakr serve`)

Rendering wants a GPU box; looking at the results does not. `moviemakr serve`
puts the workspace on a web page — scripts and their per-scene state, movies
that play and download in the browser, logs, drafts, and asset upload — so you
stop sftp'ing into the render machine.

```bash
pip install 'moviemakr[web]'
moviemakr serve --host 0.0.0.0            # then reach it over your tailnet
tailscale serve --bg 8765                 # …or with HTTPS and a real hostname
```

Put it behind `tailscale serve`, not Funnel: tailnet-only *is* the auth model,
and the app hands out workspace files.

It is deliberately **read-only toward rendering** — no start, no cancel, no job
queue. Renders stay on the CLI over ssh, where the terminal already does a
better job. What the page adds is everything around them.

A script page shows how far its render got: a bar counted in scenes, the scene
in flight, and roughly how long the rest will take at the pace of the scenes
that finished. It is read out of the run's own artefacts — `state.json` for what
completed, the newest log file for what is happening now — so it works for a
render started from any terminal, and it needs nothing running on the render
box but the render. The scene table has a `live` toggle that polls every five
seconds, and it arms itself when the page loads mid-render, so you can watch
from the sofa. Nothing there is precise: a scene is several passes and only one
of them reports steps, so every estimate is prefixed with `~`.

Uploading matters more than it looks: reference images have to live under
`assets/` for the container to see them, so being able to push a photo there
from your phone is what makes drafting away from the desk possible. Uploads are
resized to 1280px on the long edge by default, since the workspace is a git repo.

### Drafts

`drafts/*.md` are plain prose kept apart from finished scripts — who is in the
scene, the beats, the mood, what each reference is for. Write one on your phone;
turn it into a script later.

The web app does not expand drafts, because it cannot: `h3-prompt-writing` is an
instruction-only skill, so an agent has to do the work. Each draft page shows the
command:

```bash
claude "expand drafts/beach-picnic.md into a moviemakr script using h3-prompt-writing"
```

## Setting up a workspace

A fresh one is four directories and its own git repo:

```bash
WS=~/moviemakr-workspace
mkdir -p "$WS"/{scripts,assets,drafts,renders}
printf 'renders/\n.cache/\n' > "$WS/.gitignore"
git -C "$WS" init && git -C "$WS" add -A && git -C "$WS" commit -m 'workspace'

echo 'export MOVIEMAKR_WORKSPACE='"$WS" >> ~/.zshrc
```

Keep it under version control separately from this checkout: the scripts and the
reference images are the work, and `renders/` is reproducible from them.

Workspaces are freely relocatable, and you can have several. Moving one does not
invalidate a single rendered scene: fingerprints are built from container-side
paths and reference *content*, so the three `-v` mounts change while the
`/assets/…` and `/out/…` paths the model sees stay identical. Confirm with:

```bash
moviemakr render "$WS/scripts/<something>.yaml" --dry-run
moviemakr status "$WS/scripts/<something>.yaml"   # rendered scenes: up to date
```

## Notes

- **Assembly re-encodes deliberately.** The generator writes PCM audio into WebM,
  which is off-spec and does not stream-copy reliably, and scenes may differ in
  size. Each clip is normalized to uniform codecs, resolution, and frame rate
  first; the concat itself is then a cheap stream copy. Clips smaller than the
  target resolution are scaled and padded, not stretched.
- **Durations are measured, not assumed.** The model rounds frame counts up — a
  request for 86 frames produced 90 — so lengths are read back with `ffprobe`.
- **The container drops root, but must join the GPU groups** (`sdcpp`). It runs as
  `--user $(id -u):$(id -g)` so clips are owned by you rather than root, and adds
  `--group-add` for whichever groups own the device nodes (here `video` and
  `render`). Without those groups Vulkan enumerates nothing and ggml silently
  falls back to CPU. Set `docker.run_as_current_user: false` to run as root.
- **A GPU preflight runs before every render** (`sdcpp`). It calls `--list-devices` in the
  container and aborts if no accelerator is visible, because the CPU fallback is
  silent and costs hours per scene. `--allow-cpu` overrides it. If the container
  fails to start at all, the preflight reports that error rather than blaming the
  GPU. If you ever see `VRAM 0.00MB` in a log, that run is on CPU.
- **Sizes get aligned by the model.** Dimensions round up to a multiple of 32 and
  frame counts to the model's own step (540x960x120 became 544x960x124). Pick
  multiples of 32 if you want to know the exact output size in advance.
- **Model load dominates each scene** (`sdcpp`). `sd-cli` is one-shot, so roughly
  39 GB of weights reload on every invocation and there is no way to keep them
  resident across prompts. `--mmap` is in the example's `extra_args` for exactly
  this reason: with enough RAM the page cache keeps the weights warm between
  runs. The per-scene timings in the summary show how large that fixed cost
  really is. Avoiding this reload is most of why the comfy backend renders the
  same scene in roughly a third of the time.
- **`/history` is empty while a prompt runs** (`comfy`). An entry appears only
  once it finishes, so "absent from history" is the normal working state, and it
  is the queue that separates still-rendering from dropped.

## Requirements

ffmpeg/ffprobe on PATH, and Python 3.11+ with PyYAML. The engine adds its own:
`sdcpp` needs Docker and the weights on disk, `comfy` needs a reachable ComfyUI
already serving the named models. `moviemakr serve` adds the `web` extra
(FastAPI, uvicorn, Jinja2); the rest of the tool works without it.

`./moviemakr.py` is a launcher for the `moviemakr/` package next to it, so it
runs straight from a checkout with no install. `pip install -e .` additionally
provides a `moviemakr` command; `python -m moviemakr` works either way.

## Development

```bash
uv sync --extra web --extra dev
.venv/bin/pytest
.venv/bin/ruff check .
```

`uv.lock` pins the whole environment, ruff included; CI runs `uv sync --locked`,
so a change to `pyproject.toml` without a matching `uv lock` fails there rather
than giving everyone a different resolution.

The suite needs no Docker, GPU, ffmpeg, or ComfyUI: every path reaches `sd-cli`
through `to_container`, so commands and fingerprints are built entirely from
container-side paths and can be checked in a temp directory, and the comfy tests
assert on the graph and the parsing of canned server responses rather than on a
live server. The web tests hold that line too — ffmpeg is stubbed, and the route
tests `importorskip` FastAPI, so `pytest PyYAML` alone still runs everything
else. [`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs it on
3.11–3.14, plus a job with the `web` extra deliberately absent to keep FastAPI
from leaking out of `web/app.py`.

`tests/test_fingerprint.py` holds golden hashes, and `tests/test_comfy_graph.py`
a golden graph. A scene is skipped when its stored fingerprint matches, so a
change to either means every scene of every existing run re-renders — treat a
failure as a regression, not a value to bless.

The pipeline is one module per stage, in dependency order:

| module | holds |
| --- | --- |
| `errors.py` | `ConfigError`, unknown-key checking with suggestions |
| `report.py` | duration and summary formatting |
| `state.py` | reading and writing `state.json` |
| `layout.py` | `Workspace`, `RunLayout` — every path, and host ↔ container mapping |
| `media.py` | ffprobe/ffmpeg runners plus pure command builders |
| `config.py` | `SceneSettings`, `Scene`, `Script`, `load_script` |
| `backends/__init__.py` | the `backend:` registry; resolves a name to a module |
| `backends/sdcpp.py` | `sd_args`, `fingerprint`, `docker_argv`, GPU preflight |
| `backends/comfy.py` | `build_graph`, `fingerprint`, the HTTP client, server preflight |
| `assemble.py` | normalize → concat → optional music mix |
| `render.py` | `RenderOptions` and the render loop |
| `status.py` | `scene_rows` — per-scene state, shared by `status` and the web view |
| `cli.py` | argument parsing and the command bodies |
| `web/` | the optional HTTP view; only `web/app.py` imports FastAPI |

Adding a scene setting means adding a field to `SceneSettings` and a coercer to
`_COERCERS` in `config.py`; unknown keys are rejected, so a setting that is not
declared there cannot be used in a script.
