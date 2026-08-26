# moviemakr

Renders a multi-scene movie from a YAML script using `stable-diffusion.cpp`
(MiniMax-H3 Ref2VA) in Docker, one scene per invocation, then stitches the clips
into a single movie with ffmpeg.

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

Point at it with `--workspace PATH` or `$MOVIEMAKR_WORKSPACE`. With neither, the
checkout itself is used, which is how this used to work.

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
| `--allow-cpu` | skip the GPU preflight and render anyway |

Start with `--dry-run`. It prints the exact `docker run` command for every scene,
including reference-image wiring, without spending GPU time — and writes nothing,
not even the extracted frames for a reference video.

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

## Writing a script

See [`examples/example.yaml`](examples/example.yaml) for a fully commented template.
The shape is: shared `model` paths, `docker` settings, `defaults` that apply to
every scene, `continuity` and `output` policy, then an ordered `scenes` list.

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

Two independent mechanisms, both using the model's native `--ref-image`
conditioning:

- **`continuity.anchors`** — reference images under `assets/` passed to *every*
  scene. Use this for a character sheet, so the cast looks the same throughout.
- **`chain_from_previous: true`** — after each scene renders, its final frame is
  extracted to `renders/<name>/frames/` and handed to the next scene as a
  reference. This gives shot-to-shot flow. Set it globally under `continuity` or
  per scene.

They compose: a scene can use anchors, the chained frame, and its own
`ref_images` at once.

### Using a video as the reference

This model is Ref2VA, so it takes video references natively. A clip carries both
appearance and motion, which a single still cannot.

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
checked when the script loads rather than part-way through the render.

## Resume and retries

Renders are expensive, so a scene is only re-run when it actually needs to be.
Each scene is fingerprinted over its resolved command line *and the content of its
reference images*. A scene is skipped when its clip exists and the fingerprint
matches, so:

- Editing one prompt re-renders that scene only.
- If a chained scene's upstream clip changes, the chained frame's content changes
  too, and the downstream scene correctly re-renders as well.
- Ctrl-C discards the partial clip; the next run picks up from that scene.

Each scene runs in a container named `moviemakr-<script>-<scene>`, so `docker ps`
tells you which scene is rendering. Interrupting stops that container explicitly
— `docker run` is only a client, and killing it would otherwise leave the job
running under the daemon, burning CPU and RAM with nothing watching it. If you
ever suspect a stray render:

```bash
docker ps --filter name=moviemakr        # what is actually running
docker rm -f <name>                      # stop one
```

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

`scenes/` is always `.webm` because that is what `sd-cli` writes; only
`normalized/` and the finished movie follow `output.container`.

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
better job. What the page adds is everything around them. The scene table has a
`live` toggle that polls while an ssh-launched render is running, so you can
watch progress from the sofa.

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

## Moving your data into a workspace

If you have been running with everything inside the checkout:

```bash
WS=~/moviemakr-workspace
mkdir -p "$WS"/{scripts,assets,drafts,renders}
mv scripts/* assets/* "$WS"/scripts/ "$WS"/assets/   # adjust per directory
printf 'renders/\n.cache/\n' > "$WS/.gitignore"
git -C "$WS" init && git -C "$WS" add -A && git -C "$WS" commit -m 'workspace'

echo 'export MOVIEMAKR_WORKSPACE='"$WS" >> ~/.zshrc
```

Then confirm nothing moved that should not have:

```bash
moviemakr render "$WS/scripts/<something>.yaml" --dry-run
```

The three `-v` mounts should now point into the workspace while the
container-side `/assets/…` and `/out/…` paths stay exactly as before. They will:
fingerprints are built from container paths and reference *content*, so
relocating a workspace does not invalidate a single scene.

## Notes

- **Assembly re-encodes deliberately.** The generator writes PCM audio into WebM,
  which is off-spec and does not stream-copy reliably, and scenes may differ in
  size. Each clip is normalized to uniform codecs, resolution, and frame rate
  first; the concat itself is then a cheap stream copy. Clips smaller than the
  target resolution are scaled and padded, not stretched.
- **Durations are measured, not assumed.** The model rounds frame counts up — a
  request for 86 frames produced 90 — so lengths are read back with `ffprobe`.
- **The container drops root, but must join the GPU groups.** It runs as
  `--user $(id -u):$(id -g)` so clips are owned by you rather than root, and adds
  `--group-add` for whichever groups own the device nodes (here `video` and
  `render`). Without those groups Vulkan enumerates nothing and ggml silently
  falls back to CPU. Set `docker.run_as_current_user: false` to run as root.
- **A GPU preflight runs before every render.** It calls `--list-devices` in the
  container and aborts if no accelerator is visible, because the CPU fallback is
  silent and costs hours per scene. `--allow-cpu` overrides it. If the container
  fails to start at all, the preflight reports that error rather than blaming the
  GPU. If you ever see `VRAM 0.00MB` in a log, that run is on CPU.
- **Sizes get aligned by the model.** Dimensions round up to a multiple of 32 and
  frame counts to the model's own step (540x960x120 became 544x960x124). Pick
  multiples of 32 if you want to know the exact output size in advance.
- **Model load dominates each scene.** `sd-cli` is one-shot, so roughly 39 GB of
  weights reload on every invocation and there is no way to keep them resident
  across prompts. `--mmap` is in the example's `extra_args` for exactly this
  reason: with enough RAM the page cache keeps the weights warm between runs. The
  per-scene timings in the summary show how large that fixed cost really is.

## Requirements

Docker, ffmpeg/ffprobe, Python 3.11+ with PyYAML. `moviemakr serve` adds the
`web` extra (FastAPI, uvicorn, Jinja2); the rest of the tool works without it.

`./moviemakr.py` is a launcher for the `moviemakr/` package next to it, so it
runs straight from a checkout with no install. `pip install -e .` additionally
provides a `moviemakr` command; `python -m moviemakr` works either way.

## Development

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e '.[web,dev]'
.venv/bin/python -m pytest
```

The suite needs no Docker, GPU, or ffmpeg: every path reaches `sd-cli` through
`to_container`, so commands and fingerprints are built entirely from
container-side paths and can be checked in a temp directory. The web tests hold
that line too — ffmpeg is stubbed, and the route tests `importorskip` FastAPI,
so `pytest PyYAML` alone still runs everything else.

`tests/test_fingerprint.py` holds golden hashes. A scene is skipped when its
stored fingerprint matches, so a change there means every scene of every existing
run re-renders — treat a failure as a regression, not a value to bless.

The pipeline is one module per stage, in dependency order:

| module | holds |
| --- | --- |
| `errors.py` | `ConfigError`, unknown-key checking with suggestions |
| `report.py` | duration and summary formatting |
| `state.py` | reading and writing `state.json` |
| `layout.py` | `Workspace`, `RunLayout` — every path, and host ↔ container mapping |
| `media.py` | ffprobe/ffmpeg runners plus pure command builders |
| `config.py` | `SceneSettings`, `Scene`, `Script`, `load_script` |
| `docker.py` | `sd_args`, `fingerprint`, `docker_argv`, GPU preflight |
| `assemble.py` | normalize → concat → optional music mix |
| `render.py` | `RenderOptions` and the render loop |
| `status.py` | `scene_rows` — per-scene state, shared by `status` and the web view |
| `cli.py` | argument parsing and the command bodies |
| `web/` | the optional HTTP view; only `web/app.py` imports FastAPI |

Adding a scene setting means adding a field to `SceneSettings` and a coercer to
`_COERCERS` in `config.py`; unknown keys are rejected, so a setting that is not
declared there cannot be used in a script.
