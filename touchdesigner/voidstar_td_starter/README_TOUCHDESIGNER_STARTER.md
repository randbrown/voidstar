# Voidstar x TouchDesigner Starter

This folder gives you a practical starting point to combine your Voidstar scripts with a TouchDesigner live workflow.

## What is included

- `build_voidstar_project.py`: TouchDesigner Text DAT script that generates a starter network in `/project1`.
- `build_blackhole_project.py`: TouchDesigner Text DAT script that generates an audio-reactive 3D black hole simulation network in `/project1/blackhole_live`.
- `voidstar_post_bridge.py`: Offline post-process bridge that wraps your existing scripts:
  - `dvd_logo/voidstar_dvd_logo.py`
  - `glitchfield/glitchfield.py`
  - `reels_cv_overlay/reels_cv_overlay.py`
  - `title_hook/voidstar_title_hook.py`
- `requirements_voidstar_td.txt`: Python dependencies for post-processing bridge targets.
- `setup_python_env.sh`: Non-interactive bootstrap script that creates and installs a dedicated venv.

## Python environment (non-interactive)

Run this once from workspace root:

```bash
bash touchdesigner/voidstar_td_starter/setup_python_env.sh
```

What it does:
- Creates `.venv_voidstar_td` in the repo root (if missing).
- Installs required packages without interactive prompts.
- Prints the Python path to use in TouchDesigner external Python settings.

If you want a different Python binary for creating the venv:

```bash
PYTHON_BIN=python3.11 bash touchdesigner/voidstar_td_starter/setup_python_env.sh
```

## Important: .toe creation

A `.toe` file is a binary TouchDesigner project file. It must be created from inside TouchDesigner.

Workflow:
1. Open TouchDesigner.
2. Create a new empty project.
3. Create a `Text DAT` in `/project1`.
4. Paste in the contents of `build_voidstar_project.py` and click **Run Script**.
5. Save project as `voidstar_starter.toe`.

If build fails, open `/project1/voidstar_build_log` and copy the full traceback text directly from that DAT.

## Black hole simulation

`build_blackhole_project.py` creates `/project1/blackhole_live`, a self-contained 3D scene:

| Operator | Role |
|---|---|
| `geo_event_horizon` | Black sphere — radius pulses with audio amplitude |
| `geo_disc` | Torus SOP with animated noise turbulence — colour brightens orange→white-hot |
| `geo_particles` | Circle SOP ring (240 pts) with spiral noise — rendered as point sprites |
| `cam` | Elevated angled camera (Ty 7, Tz 14, Rx −26) |
| `light_key` | Warm key light (off-axis upper right) |
| `light_fill` | Cool fill light (opposite side, low intensity) |
| `render1` | 1920×1080 render, black background |
| `feedback1` + `comp_fb` | Trail accumulation loop — decay rate driven by audio |
| `displace_lens` | Gravitational lensing warp via animated `noise_warp` |
| `blur_glow` | Bloom corona — radius driven by audio |
| `out1` | Final output |

**Audio routing** (same switch pattern as the starter):

```
audio_loopback  ─┐
                  ├→ audio_live_merge → audio_src_switch[0]
audio_instrument ─┘                                        │
                                                           ├→ analyze → math → filter → audio_env
audio_file ─────────────────────────→ audio_src_switch[1] ─┘
```

Set `audio_src_switch.par.Index = 0` for live, `1` for file.  Run `audio_diag` DAT to verify levels.

**Kinect / hand interaction** — not wired by default.  Add a `kinectCHOP` and route hand position channels into `cam.par.Tx.expr` / `cam.par.Ty.expr` to move the viewpoint with your hand.

## Audio reactivity architecture (robust for your goals)

Use two audio inputs in TouchDesigner and merge them:
- `audio_instrument`: your guitar/interface input.
- `audio_loopback`: system loopback feed (for Strudel/Hydra/browser audio).

Why this is robust:
- You can perform live with hardware input and software audio at the same time.
- You can mute/solo each source at CHOP level.
- One shared envelope channel can drive all visual controls.

### Do you still need Cable Audio / virtual routing?

Usually yes for software-generated audio (Strudel/Hydra), because TD needs a device endpoint to capture system/browser output reliably.

Recommended by platform:
- Linux: PipeWire or JACK loopback (Helvum/Patchance/qpwgraph) plus your interface as the main clock.
- Windows: ASIO interface for instruments + WASAPI loopback or virtual cable for browser audio.
- macOS: BlackHole or Loopback for system audio capture.

If Strudel/Hydra run on the same machine as TouchDesigner, loopback is the most dependable method.

## Real-time vs post strategy

- Live: Use `/project1/voidstar_live` only, with native TOP/CHOP effects for lowest latency.
- Post: Export clip, then run `voidstar_post_bridge.py` to apply heavier OpenCV/FFmpeg scripts.

This split keeps performance stable while preserving your full Voidstar look in final renders.

## Suggested first session

1. In `voidstar_live/src_switch`, pick camera or movie input.
2. In `voidstar_live/audio_instrument`, select your audio interface channels.
3. In `voidstar_live/audio_loopback`, select your loopback source.
4. Watch `voidstar_live/audio_env` and verify it responds.
5. Run `audio_diag` DAT in `/project1/voidstar_live` to print a green/red status snapshot for instrument and loopback inputs.
5. Tweak these visual nodes:
   - `displace1` for distortion amount.
   - `blur1` for smear intensity.
   - `feedback1`/`transform_fb` for motion trails.
   - `dvdlogo_png` for your logo source.
   - `title_hook` for overlay text.
6. Save `.toe` once stable.

## Post-process examples

### DVD logo pass

```bash
.venv_voidstar_td/bin/python touchdesigner/voidstar_td_starter/voidstar_post_bridge.py \
  --effect dvdlogo \
  --input /absolute/path/live_export.mp4 \
  --output /absolute/path/live_export_dvd.mp4 \
  --logo dvd_logo/voidstar_logo_0.png
```

### Glitchfield pass

```bash
.venv_voidstar_td/bin/python touchdesigner/voidstar_td_starter/voidstar_post_bridge.py \
  --effect glitchfield \
  --input /absolute/path/live_export.mp4 \
  --output /absolute/path/live_export_glitch.mp4 \
  --glitch-effect combo
```

### Title hook pass

```bash
.venv_voidstar_td/bin/python touchdesigner/voidstar_td_starter/voidstar_post_bridge.py \
  --effect title_hook \
  --input /absolute/path/live_export.mp4 \
  --output /absolute/path/live_export_title.mp4 \
  --title "VOIDSTAR" \
  --secondary-text "LIVE SIGNAL"
```

## Notes

- Your existing scripts are currently optimized for file-based processing, not direct per-frame TOP callbacks.
- If you want, next step is to refactor one effect (best first candidate: `dvdlogo`) into a Script TOP-compatible module for true in-TD real-time behavior.
- `run_post_from_td_dat.py` now prefers `.venv_voidstar_td/bin/python` automatically when present.
