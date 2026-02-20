#!/usr/bin/env python3
import argparse, math, time
from pathlib import Path

import cv2
import numpy as np

# -------------------------
# util / logging
# -------------------------
def log(msg: str) -> None:
    print(f"[voidstar] {msg}", flush=True)

def fmt_float_token(x: float, ndigits: int = 3) -> str:
    return f"{x:.{ndigits}f}".replace(".", "p")

def default_out_path(left_path: str, right_path: str, seconds: float, fps: int,
                     cx: float, cy: float, rad: float, w: int, h: int) -> str:
    L, R = Path(left_path), Path(right_path)
    out = (
        f"{L.stem}_flowmorph_"
        f"s{fmt_float_token(seconds,2)}_fps{fps}_"
        f"cx{fmt_float_token(cx,3)}_cy{fmt_float_token(cy,3)}_"
        f"rad{fmt_float_token(rad,3)}_{w}x{h}_"
        f"R{R.stem}.mp4"
    )
    return str(L.parent / out)

def smooth_pingpong(t: float) -> float:
    # loop-safe 0->1->0 across [0,1)
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * t)

# -------------------------
# warping helpers
# -------------------------
def flow_warp(img: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
    """
    Warp img by fraction t of full flow field.
    flow is (H,W,2) mapping A->B (dx,dy).
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mapx = (xx + flow[..., 0] * t).astype(np.float32)
    mapy = (yy + flow[..., 1] * t).astype(np.float32)
    return cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def compute_flow_dis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Fast CPU optical flow. Surprisingly good for stylized imagery.
    """
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # DIS: fast + decent. tweak preset for quality
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    flow = dis.calc(a_g, b_g, None)  # (H,W,2) float32
    return flow

# Optional: RAFT via torch/torchvision if installed.
def compute_flow_raft(a: np.ndarray, b: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    Requires: torch + torchvision with RAFT weights available.
    If this fails, we fallback to DIS automatically.
    """
    import torch
    import torchvision
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    # RAFT expects float tensor [B,3,H,W] in 0..1
    def to_tensor(x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(x).permute(2,0,1).float() / 255.0
        return t.unsqueeze(0)

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    preprocess = weights.transforms()

    ta = preprocess(to_tensor(a).to(device))
    tb = preprocess(to_tensor(b).to(device))

    with torch.no_grad():
        # output is list of flows at different scales; last is highest res
        flows = model(ta, tb)
        flow = flows[-1][0].permute(1,2,0).detach().float().cpu().numpy()  # (H,W,2)

    return flow

# -------------------------
# your vibe effects (kept simple)
# -------------------------
def subtle_drift(img: np.ndarray, t: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w*0.5, h*0.5
    angle = 0.25 * math.sin(2*math.pi*t)
    scale = 1.0 + 0.004 * math.sin(2*math.pi*(t+0.13))
    dx = 2.0 * math.sin(2*math.pi*(t+0.21))
    dy = 2.0 * math.sin(2*math.pi*(t+0.47))
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0,2] += dx; M[1,2] += dy
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

def vortex_swirl(frame: np.ndarray, t: float, center_xy, radius_px: float) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = center_xy
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx*dx + dy*dy) + 1e-6
    m = np.clip(1.0 - (r / radius_px), 0.0, 1.0)
    m = m*m*(3.0 - 2.0*m)

    swirl = 1.0 * math.sin(2*math.pi*t)
    dtheta = swirl * m
    cosv = np.cos(dtheta); sinv = np.sin(dtheta)
    x2 = cx + dx*cosv - dy*sinv
    y2 = cy + dx*sinv + dy*cosv
    out = cv2.remap(frame, x2.astype(np.float32), y2.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    glow = 0.10 + 0.10*(0.5 - 0.5*math.cos(2*math.pi*t))
    blurred = cv2.GaussianBlur(out, (0,0), sigmaX=6.0, sigmaY=6.0)
    out = cv2.addWeighted(out, 1.0, blurred, glow, 0)
    return out

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("-o","--out", default=None)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--center-x", type=float, default=0.5)
    ap.add_argument("--center-y", type=float, default=0.5)
    ap.add_argument("--swirl-radius", type=float, default=0.22)
    ap.add_argument("--status-every", type=int, default=30)
    ap.add_argument("--flow", choices=["raft","dis"], default="raft")
    args = ap.parse_args()

    t0 = time.perf_counter()

    A = cv2.imread(args.left, cv2.IMREAD_COLOR)
    B = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if A is None or B is None:
        raise SystemExit("[voidstar] ERROR: failed to load input images")

    h = min(A.shape[0], B.shape[0])
    w = min(A.shape[1], B.shape[1])
    A = cv2.resize(A, (w,h), interpolation=cv2.INTER_AREA)
    B = cv2.resize(B, (w,h), interpolation=cv2.INTER_AREA)

    frames = int(round(args.seconds * args.fps))
    if frames < 2:
        raise SystemExit("[voidstar] ERROR: seconds*fps must be >= 2 frames")

    out_path = args.out or default_out_path(args.left, args.right, args.seconds, args.fps,
                                           args.center_x, args.center_y, args.swirl_radius, w, h)

    cx = args.center_x * w
    cy = args.center_y * h
    radius_px = args.swirl_radius * min(w,h)

    log("starting flow morph")
    log(f"left={args.left}")
    log(f"right={args.right}")
    log(f"size={w}x{h} fps={args.fps} seconds={args.seconds:.3f} frames={frames}")
    log(f"vortex center=({args.center_x:.3f},{args.center_y:.3f}) radius={args.swirl_radius:.3f} (px={radius_px:.1f})")
    log(f"flow={args.flow}")
    log(f"out={out_path}")

    # compute flow fields once (big win)
    flow_t0 = time.perf_counter()
    flow_ab = None
    flow_ba = None

    if args.flow == "raft":
        try:
            log("computing RAFT flow (GPU)…")
            flow_ab = compute_flow_raft(A, B, device="cuda")
            flow_ba = compute_flow_raft(B, A, device="cuda")
            log(f"RAFT flow ok in {time.perf_counter() - flow_t0:.2f}s")
        except Exception as e:
            log(f"RAFT failed ({e}); falling back to DIS (CPU)")
            args.flow = "dis"

    if args.flow == "dis":
        log("computing DIS flow (CPU)…")
        flow_ab = compute_flow_dis(A, B)
        flow_ba = compute_flow_dis(B, A)
        log(f"DIS flow ok in {time.perf_counter() - flow_t0:.2f}s")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, args.fps, (w,h))
    if not vw.isOpened():
        raise SystemExit("[voidstar] ERROR: failed to open VideoWriter (mp4v)")

    t_write0 = time.perf_counter()
    for i in range(frames):
        t = i / frames  # loop-safe
        mix = smooth_pingpong(t)  # 0..1..0

        # drift each endpoint a little (keeps network "alive")
        A2 = subtle_drift(A, t)
        B2 = subtle_drift(B, (t+0.37) % 1.0)

        # warp A forward towards B, and B backward towards A
        # When mix=0: show A. When mix=1: show B.
        Aw = flow_warp(A2, flow_ab, mix)
        Bw = flow_warp(B2, flow_ba, 1.0 - mix)

        frame = cv2.addWeighted(Aw, 1.0 - mix, Bw, mix, 0.0)

        # vortex “processing” swirl
        frame = vortex_swirl(frame, t, (cx, cy), radius_px)

        vw.write(frame)

        if args.status_every > 0 and (i % args.status_every == 0 or i == frames-1):
            done = i + 1
            elapsed = time.perf_counter() - t_write0
            fps_eff = done / elapsed if elapsed > 1e-9 else 0.0
            rem = frames - done
            eta = rem / fps_eff if fps_eff > 1e-9 else float("inf")
            pct = 100.0 * done / frames
            log(f"frames={done}/{frames} ({pct:5.1f}%) speed={fps_eff:6.2f} fps ETA={eta:6.1f}s")

    vw.release()
    log(f"done in {time.perf_counter()-t0:.2f}s -> {out_path}")

if __name__ == "__main__":
    main()
