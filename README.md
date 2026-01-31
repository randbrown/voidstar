# voidstar

**voidstar** is a personal toolkit of audio, video, and visual processing tools used in live performance, performance art, and experimental media work.

This repository collects scripts, utilities, and small pipelines that sit at the intersection of:

- live music and improvisation
- computer vision and motion tracking
- audio–visual synchronization
- generative and algorithmic visuals
- post-processing for performance documentation (reels, clips, archives)

The focus is on **practical tools** that are actually used in real performances and recordings, not polished consumer software.

---

## Philosophy

- Tools over products  
- Deterministic pipelines over “magic”  
- Minimal defaults, expressive opt-in features  
- Designed for *use*, not just demos  
- Optimized for iteration and experimentation  

Most scripts start simple, then grow features organically as performance needs evolve.

---

## Contents

This repo may include (but is not limited to):

- Pose and motion–based video overlays
- Computer vision experiments
- Audio–video alignment and processing utilities
- Reel / short-form video generation tools
- Performance visualization experiments
- One-off scripts that proved useful enough to keep

Each major tool or script is generally self-contained and documented locally.

---

## Example: Pose Overlay Tools

One component of this repository is a pose-based overlay system for vertical video, used to visualize motion and performance energy in recorded clips.

Features may include:
- MediaPipe pose tracking
- Deterministic CFR video processing
- Optional motion trails
- Optional velocity-based visualization
- Minimal, diagrammatic overlays by default

(See individual tool directories or scripts for details.)

---

## Requirements

Most tools in this repo rely on:

- Python 3.10+
- ffmpeg (installed system-wide)
- Common scientific / media libraries (OpenCV, NumPy, etc.)

Each script documents its own dependencies where applicable.

---

## Structure

This repository is intentionally **not** a monolithic application.

Expect:
- standalone scripts
- small focused modules
- evolving structure over time

Stability is maintained per-tool, not necessarily across the entire repo.

---

## Status

This is an **active, evolving workspace**.

Things may change, improve, or be replaced as techniques, performances, and requirements evolve.  
Baselines may be tagged when useful, but experimentation is encouraged.

---

## License

Personal / experimental use.  
If you’re interested in reuse, adaptation, or collaboration, reach out.

---

## Name

> *voidstar* — a dense center of collapse and emergence  
> where sound, motion, and light orbit briefly before dispersing
