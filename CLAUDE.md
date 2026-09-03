# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`moviemakr` renders a multi-scene movie from a YAML script. The model only
produces a few seconds of video per invocation, so the tool renders scenes in
order, resumes where it left off, and stitches the clips into a single movie with
ffmpeg.

Two engines render a scene, chosen by the script's `backend:` key:

- **`sdcpp`** (default) — one `docker run` of `sd-cli` (stable-diffusion.cpp,
  MiniMax-H3 Ref2VA) in `vid_gen` mode.
- **`comfy`** — an API graph submitted to a running ComfyUI. Slower to set up,
  but it can anchor a whole *segment* of the previous scene, audio included, which
  sd-cli cannot express.

The code lives in the `moviemakr/` package; `moviemakr.py` at the root is a
four-line launcher so `./moviemakr.py …` keeps working without installing
anything. Dependencies: Docker, ffmpeg/ffprobe on PATH, Python 3.11+ with PyYAML.
`moviemakr serve` additionally needs the `web` extra.

## The workspace (code and data are separate)

The **data** — `scripts/`, `assets/`, `drafts/`, `renders/` — lives in a
*workspace* directory that is not the code checkout. **None of it is in this
repo**, and the checkout is not a fallback workspace. `Workspace` (in
`layout.py`) owns that split, and `Workspace.resolve` picks the root from
exactly two sources:

1. `--workspace PATH`
2. `$MOVIEMAKR_WORKSPACE`

With neither, it raises `no workspace: pass --workspace or set
MOVIEMAKR_WORKSPACE`. There is deliberately no third fallback — the checkout
used to be one, which meant a forgotten workspace resolved to a valid directory
and either failed later, confusingly, or quietly wrote a new `assets/` into the
code tree (`stills`, and the web uploader).

`config.py` reads `workspace.assets_dir` and `workspace.renders_dir` and nothing
else; `RunLayout.build` already took its three mount bases explicitly, so it did
not change. Moving a workspace does **not** change any fingerprint: `sd_args`
emits only container-side paths, and refs are hashed by content.

## Commands

```bash
export MOVIEMAKR_WORKSPACE=~/moviemakr-workspace
cd "$MOVIEMAKR_WORKSPACE"                          # script paths are just paths

moviemakr render   scripts/h3/beach.yaml --dry-run  # print docker commands, render nothing
moviemakr render   scripts/h3/beach.yaml            # render all scenes, then assemble
moviemakr status   scripts/h3/beach.yaml            # per-scene state + measured durations
moviemakr assemble scripts/h3/beach.yaml            # re-stitch from existing clips
moviemakr stills   scripts/h3/beach.yaml --count 6  # pull reference stills from a clip
moviemakr serve --host 0.0.0.0                      # browse the workspace over HTTP
```

`python -m moviemakr` and the installed `moviemakr` script are equivalent entry
points. Script paths are ordinary CLI arguments and are **not** resolved against
the workspace — `--workspace` only decides where assets and renders live.

`stills` extracts evenly spaced frames from one rendered scene clip
(`--scene ID`, or the only scene) and writes them to `assets/` by default —
because a reference *image* has to live under that mount to survive
`to_container()`. It is how a turnaround render becomes reusable character
references; see `<workspace>/scripts/h3/niulai-cat-sheet.yaml`. Timestamps are segment
midpoints, never the first or last frame, which are the blurriest.

### Tests

```bash
.venv/bin/python -m pytest          # ~535 tests, under a second
```

The suite is hermetic — no Docker, GPU, or ffmpeg. That works because `sd_args`
maps every path through `to_container` before it reaches the command line, so
the argv (and therefore the fingerprint) contains only container-side paths.
The web tests keep that property: ffmpeg is monkeypatched in
`tests/test_web_assets.py`, and the `web_workspace` fixture stores a probe in
`state.json` so the scene table never reaches ffprobe.

`tests/test_web_routes.py` `importorskip`s FastAPI and httpx, so the suite still
runs with nothing but pytest and PyYAML — it just covers less. Install the
extras to get the route coverage:

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e '.[web,dev]'
```

There is no pip or ensurepip on this box; the venv was made with
`uv venv .venv && uv pip install --python .venv/bin/python pytest PyYAML`.

**`tests/test_fingerprint.py` and `tests/test_sd_args.py` are the anchor.** Their
golden values were captured from the pre-package single-file version. A scene is
skipped when its stored fingerprint matches, so if the hash drifts, every scene
of every existing run silently re-renders at hours apiece. Treat a golden failure
as a real regression, not a value to update — unless you have deliberately
changed `sd_args` and accept re-rendering everything.

Selective/iterative rendering: `--only 2,4-6` (by index), `--scene kitchen,feast`
(by id), `--force` (re-render even if up to date), `--retries N` (default 2),
`--halt-on-failure`, `--no-assemble`, `--allow-cpu` (skip the GPU preflight).

Always start a new script with `--dry-run` — it prints the exact `docker run` for
every scene without spending GPU time. Rendering a real scene costs minutes to
hours (≈39 GB of weights reload per invocation), so avoid triggering actual
renders while developing; use `--dry-run` to validate changes to command
construction.

## Architecture

Still a linear pipeline, now one module per stage. The import graph is acyclic
and the four leaves at the top depend on nothing but stdlib and `errors`:

| module | holds | imports |
| --- | --- | --- |
| `errors.py` | `ConfigError`, `check_keys`, `suggest` | stdlib |
| `report.py` | `fmt_duration`, `format_summary` (pure) | stdlib |
| `state.py` | `load_state` / `save_state`, taking a `Path` | stdlib |
| `layout.py` | `slugify`, `Workspace`, `RunLayout` (incl. `to_container`) | errors |
| `media.py` | codec table, ffprobe/ffmpeg runners + pure command builders | errors |
| `config.py` | `SceneSettings`, `Scene`, `Script`, `load_script` | errors, layout, media |
| `backends/__init__.py` | the `backend:` registry, `check_name`, `resolve` | errors |
| `backends/sdcpp.py` | `sd_args`, `fingerprint`, `docker_argv`, `check_gpu`, `run_scene` | errors, layout, config |
| `backends/comfy.py` | `build_graph`, `fingerprint`, frame-grid helpers | config |
| `assemble.py` | normalize → concat → optional music mix | layout, config, media |
| `render.py` | `RenderOptions`, the render loop and its helpers | most of the above |
| `status.py` | `scene_rows` — per-scene state incl. **stale** | config, backends, render, state, media |
| `cli.py` | argparse, `cmd_status` / `cmd_assemble` / `cmd_serve` | everything |
| `web/` | the HTTP view (optional extra) | see below |

Three placements are deliberate and worth not undoing:

- **`slugify` lives in `layout.py`, not `config.py`.** `layout` needs it for the
  refvideo tag and the movie filename, and `config` builds the layout — putting
  it in `config` is the one real import cycle available here.
- **`to_container` is a `RunLayout` method**, not a docker concern. Its three
  mount bases *are* the layout's state.
- **Each backend owns its own fingerprint.** `comfy.fingerprint` hashes the API
  graph, `sdcpp.fingerprint` hashes the argv. They must not be unified: the same
  scene rendered through two engines is not the same output, so a run that
  switched backends has to re-render rather than resume. Both exclude the
  *location* of the output (container paths, `filename_prefix`) so that moving
  the workspace invalidates nothing.

- **`fingerprint` sits directly below `sd_args` in `backends/sdcpp.py`.** Their contract
  is "the hash is exactly this argv plus reference content", so they have to be
  edited together.

`media.py` must not import `config`: its runners take plain paths and a
`NormalizeSpec`, which is what keeps it testable without a `Script`.

**`status.py` exists so `cmd_status` and the web view cannot drift.** The
*stale* computation — recompute the fingerprint, compare it to the stored one —
is the answer to "what would a render actually redo", and there must be exactly
one of it. `cli.cmd_status` is now `print_summary(scene_rows(script), …)`.

### `moviemakr/web/` — the HTTP view

`moviemakr serve` browses a workspace: scripts and their per-scene state,
in-browser playback and download of movies and clips, logs, a plain-text draft
editor, and script/asset upload. Reached from a phone over Tailscale, it
replaces sftp'ing into the render box.

**It never starts a render and never calls an LLM.** Rendering stays on the CLI
over ssh, and drafts are expanded by an agent, not the server (see below).

| module | holds | needs FastAPI? |
| --- | --- | --- |
| `paths.py` | `safe_path`, `safe_stem` — the security boundary | no |
| `browse.py` | workspace → plain dicts for the templates | no |
| `assets.py` | image upload validation/resize, thumbnail cache (ffmpeg) | no |
| `scripts.py` | YAML upload validation and placement under `scripts/` | no |
| `app.py` | `create_app`, every route | **yes** |

Three things to preserve:

- **`app.py` is the only module that imports FastAPI**, and `web/__init__.py`
  defers its import. That is what lets `moviemakr render` keep working — and the
  bulk of the web tests keep running — where the `web` extra is not installed.
- **Every URL-derived path goes through `safe_path`.** It rejects absolute
  paths, `..`, and symlinks leaving the tree. Each route scopes to its own base
  dir (`scripts_dir`, `logs_dir`, `assets_dir`, …) rather than one shared
  `/media/<path>` handler that has to work out which base applies.
- **Route registration order is load-bearing.** Starlette compiles `{key:path}`
  to a greedy `.*`, so `/scripts/{key:path}/raw` and friends must be registered
  *before* the `/scripts/{key:path}` catch-all, or `/scripts/a/raw` is read as a
  script named `a/raw`.

Uploads are how a workspace stays in sync with another laptop, now that none of
the data is in this repo. The two uploaders differ on purpose:

- **An asset is never overwritten** (`unique_path` suffixes it) — a script may
  already reference the name. **A script is overwritten only with `replace`**,
  because re-uploading the finalised YAML *is* the update path; keep the
  workspace under git and that stays recoverable.
- **`scripts.py` rejects only what is not a script at all** (not UTF-8, not
  YAML, not a mapping, no `scenes`). A script that parses but fails
  `load_script` — nearly always a `ref_images` entry not uploaded yet — is
  stored with a warning, because the script usually arrives before its refs.
  The index already renders an unloadable script as an error row.

A script that fails to load is rendered as an error row, never raised — one bad
YAML must not take the index down. Polling the scene table (the one dynamic
thing on the site) is ~20 lines of inline `fetch`, not a JS framework: nothing is
vendored and there is no CDN dependency.

`starlette>=0.45` is a hard floor in the `web` extra, not a preference —
`FileResponse` only learned HTTP Range there, and without Range iOS cannot
scrub a video.

### Drafts

`<workspace>/drafts/*.md` are plain prose, deliberately kept apart from
`scripts/`: they are pre-script notes (who is in it, the beats, the mood), the
shape recorded in the header of `<workspace>/scripts/h3/josy-beach-drive.yaml`.

The web app stores and edits them but **cannot expand them** —
`h3-prompt-writing` is an instruction-only skill with no tools, so an agent has
to do it. The draft page prints the command to run at a terminal.

### Path mounting model (central to everything)

The container sees exactly three read/write-scoped mounts, and
`RunLayout.to_container()` translates any host path into its container-side
equivalent:

- `model.root` → `/models` (read-only)
- `<workspace>/assets` → `/assets` (read-only)
- `<workspace>/renders/<script-name>` → `/out` (read-write)

Any host path handed to `sd-cli` **must** live under one of these, or
`to_container()` raises. Reference *images* therefore must sit under `assets/`,
and this is checked at load time. Reference *videos* are the exception: they are
transcoded into frame directories under the run dir (`/out`), so their source
files can live anywhere.

### Resume via content fingerprinting

A scene is skipped when its clip exists and its fingerprint matches the stored one
in `renders/<name>/state.json`. The fingerprint (`fingerprint()`) hashes the
resolved `sd-cli` args **plus the byte content of every reference image**. This
content-hashing is deliberate and load-bearing for chaining: when scene N
re-renders, scene N+1's chained frame changes on disk under an unchanged path, and
hashing content (not the path) is what correctly invalidates N+1 too.

`fingerprint()` splits into a pure `digest()` over tokens, plus `ref_token()` /
`refvideo_token()` that touch the disk. The byte order is fixed — every arg
followed by a NUL, then the ref-video tokens, then the ref tokens. Note that
`docker.image`, the mounts and the container name are deliberately **not**
hashed, so changing the image invalidates nothing.

A ref that does not exist yet hashes its *host path* instead of its content.
That branch is only reachable in a dry run (a real run has already extracted the
previous frame), which is why a chained scene always prints "render" under
`--dry-run` even when it would skip for real.

`status` recomputes the fingerprint, so it reports **stale** for a scene whose
script has changed since it rendered — telling you what a render would actually
redo.

### Continuity mechanisms

sdcpp has two, both riding the model's `--ref-image`:

- `continuity.anchors` — images passed to *every* scene (e.g. a character sheet).
- `chain_from_previous` — each scene's final frame is extracted to `frames/` and
  fed to the next scene as a reference. Scenes outside a filtered selection still
  contribute their last frame, so `--only`/`--scene` runs don't break downstream
  continuity.

comfy adds a third that sd-cli cannot express, `overlap_frames` (default 22):

- The **tail of the previous clip** — video *and* its soundtrack — is anchored at
  frame 0 of the next scene through `MiniMaxH3AddGuide`, then trimmed back off at
  assembly. A single handed-off frame restarts both motion and the soundscape at
  every seam; a segment carries them across.
- **Two halves that must agree exactly.** The node snaps a guide clip *down* to
  the 17k+5 grid without saying so, so `effective_overlap` snaps once and the
  count actually anchored is written to `state.json`. Assembly trims from
  `state.json`, never from the script's current `overlap_frames` — editing the
  value must not change how yesterday's clip is cut.
- Both cuts seek *before* `-i` so video and audio move together. A filter-side
  trim would shift the picture and leave the sound where it was.
- It costs compute: at length 124 an overlap of 22 keeps 102 new frames, so about
  18% of each scene is regenerated and thrown away.

**Measured on gfx1151**, two 56-frame scenes at 640x384 with a 22-frame overlap:
7m26s under `--cache-ram 24 96` (10m25s under `--cache-none`). The anchored frames
match the previous scene at ~33.8 dB PSNR against 19.9 dB for frames the same
distance apart in ordinary motion - so the guide demonstrably reproduces the tail
rather than merely not erroring. The residual seam is a ~2-4 frame step rather
than a clean 1-frame one (33.5 dB against ~43 dB for a true adjacent pair),
because scene N+1 *regenerates* the anchor instead of copying it. Crossfading the
overlap instead of trimming it is the obvious next lever if that ever matters.

### ComfyUI gotchas (don't "simplify" these away)

- **`/history` is empty while a prompt runs.** An entry appears only when it
  finishes, so "absent from history" is the normal working state. The queue is
  what separates still-rendering from dropped — without that check the wait loop
  spins forever, which is exactly what happened the first time.
- **`SaveVideo` reports its mp4 under `images`**, not `video`. Reading `video`
  finds nothing and looks like a failed render.
- **`run_scene` returns shell-style codes** so the existing retry loop drives it
  unchanged: 0 ok, 1 execution error, 2 graph rejected, 3 unreachable, 4 nothing
  collected, 5 prompt vanished.
- **Collection prefers a file copy** over `/view`: a local ComfyUI shares its
  output directory, and pushing hundreds of MB through HTTP for no reason is
  slower and can time out. The HTTP path stays as the remote-server fallback.

### GPU / container gotchas (don't "simplify" these away)

- The container runs as `--user $(id -u):$(id -g)` and must `--group-add` the
  groups owning the GPU device nodes. Without those groups Vulkan enumerates
  nothing and ggml silently falls back to CPU (hours per scene, no error).
  `device_gids()` derives them; `docker.run_as_current_user: false` runs as root.
- A GPU preflight (`check_gpu`, `--list-devices`) runs before every real render to
  catch the silent CPU fallback in seconds. `--allow-cpu` bypasses it.
- `docker run` is only a client — killing it orphans the container under the
  daemon. Every interrupt/retry path calls `kill_container()` (`docker rm -f`) to
  reach the daemon directly. Containers are named `moviemakr-<script>-<scene>`.

### Why assembly re-encodes

The generator writes PCM audio into WebM (off-spec, doesn't stream-copy reliably)
and scenes may differ in size. `normalize_clip` re-encodes each clip to uniform
codecs/resolution/fps (scaling and padding, never stretching) so the concat itself
is a cheap stream copy. Durations are always measured with `ffprobe`, never
assumed — the model rounds `video_frames` up.

## Workspace layout

```
<workspace>/
  scripts/        the YAML, nested freely (h3/, short/, …)
  assets/         reference images - must live here to be container-reachable
  drafts/         plain-prose pre-scripts, expanded by an agent
  renders/<script-name>/
  .cache/thumbs/  web thumbnails, disposable
```

Inside a run dir: `scenes/` raw model output · `frames/` last frame per scene
(for chaining) · `normalized/` uniform intermediates for concat · `refvideos/`
extracted reference frames · `logs/` per-attempt logs · `state.json`
fingerprints/timings/probes · `<script-name>.mp4` the finished movie.

Only `renders/` and `.cache/` are disposable; keep the workspace under its own
git repo, separate from this checkout.

## Script format

See `examples/example.yaml` for the fully commented template: `model` paths,
`docker` settings, `defaults` (any key overridable per scene), `continuity`,
`output`, then an ordered `scenes` list. `style_suffix` is appended to every
prompt to keep the look consistent.

**Unknown keys are a hard error**, in every block, with a did-you-mean
suggestion. Adding a new setting therefore means adding it to `SceneSettings`
(and its coercer in `_COERCERS`) or to the relevant `*_KEYS` frozenset in
`config.py` — otherwise scripts using it will fail to load. Scene settings are a
frozen, slotted dataclass, so they are typed once at load time rather than
`int(...)`-cast at each use site.

## Prompt writing (`h3-prompt-writing` skill)

`.claude/skills/` holds MiniMax's own H3 skills, installed from
[MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) with
`npx skills add` (versions pinned in `skills-lock.json`; `npx skills update` to
refresh). Only **`h3-prompt-writing`** is usable here — the eight style
generators declare `compatibility: Requires the MiniMax Hub agent` and call
`hub_*` tools that don't exist in this project. Their `SKILL.md` and
`references/` files are still worth reading as style vocabulary, but don't run
them as workflows.

Use `h3-prompt-writing` when writing or rewriting a scene `prompt:`. This project
is Ref2VA, so follow `references/ref-en.txt` (six sections: `subject_definitions`,
`summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`) — `references/base-en.txt` covers the keyframe modes, which
moviemakr doesn't drive.

The guide's output is plain labelled prose, so it drops straight into a scene's
`prompt:` and out through `--prompt`. Three things to get right:

- **Use a literal block scalar (`|-`), not `>-`.** The folded scalar collapses the
  field labels onto one line. Existing single-sentence prompts can stay on `>-`.
- **Set `style_suffix: ""` on H3-structured scripts.** `full_prompt()`
  ([moviemakr.py:69-73](moviemakr.py#L69-L73)) appends the suffix as `. <suffix>`
  at the very end — after `non_diegetic_music:` — which reads as part of the music
  description. Put the look in `detailed_description` instead.
- **Reference labels follow the `-r` order in `sd_args`.** Refs are ordered
  chained-previous-frame first (inserted at index 0 in `render`), then
  `continuity.anchors`, then the scene's own `ref_images`; `--increase-ref-index`
  gives each its own index. So `<Picture 1>` is the previous scene's last frame on
  a chained scene, and the first anchor on a scene with `chain_from_previous:
  false`. Number the labels against the scene's actual ref list, not the YAML
  reading order. Reference *videos* arrive as `--ref-video` frame directories
  after the images.

Validate any rewrite with `--dry-run` before spending GPU time — it prints the
exact `--prompt` string that will reach `sd-cli`.
