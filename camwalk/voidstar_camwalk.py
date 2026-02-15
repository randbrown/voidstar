#!/usr/bin/env python3
"""
VoidStar CamWalk — Ultimate Musical Edition
Author: ChatGPT for Randy Brown 🌌

Major features:
- cinematic slow camera
- continuous drift (Perlin-like)
- continuous rotation option
- real audio reactive zoom
- bass reactive boost
- motion focus toward movement
- HUD overlay
- NVENC auto
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# Paths
# ============================================================

def get_username():
    return os.environ.get("USER") or os.environ.get("USERNAME") or "brown"

def default_videos_dir():
    return Path(f"/mnt/c/users/{get_username()}/Videos")

# ============================================================
# Encoder detection
# ============================================================

def detect_nvenc():
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return "h264_nvenc" in p.stdout
    except Exception:
        return False

def pick_codec_and_preset(codec_req, preset_req):
    if codec_req == "libx264":
        return "libx264", (preset_req or "veryfast")
    if codec_req == "h264_nvenc":
        return "h264_nvenc", (preset_req or "p4")
    if detect_nvenc():
        return "h264_nvenc", "p4"
    return "libx264", "veryfast"

# ============================================================
# Audio analyzer (REAL)
# ============================================================

class AudioAnalyzer:
    def __init__(self, input_path, fps, start_sec=0.0, duration_sec=0.0):
        self.sample_rate = 48000
        self.samples_per_frame = int(self.sample_rate / fps)

        ff_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
        ]
        if start_sec > 0:
            ff_cmd += ["-ss", f"{start_sec}"]
        ff_cmd += ["-i", str(input_path)]
        if duration_sec > 0:
            ff_cmd += ["-t", f"{duration_sec}"]
        ff_cmd += [
            "-vn",
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-f", "f32le", "pipe:1"
        ]

        self.proc = subprocess.Popen(
            ff_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.smooth = 0.0
        self.closed = False

    def read_level(self):
        if self.closed or not self.proc or not self.proc.stdout:
            return self.smooth

        needed = self.samples_per_frame * 4
        try:
            buf = self.proc.stdout.read(needed)
        except Exception:
            return self.smooth

        if not buf:
            return self.smooth

        audio = np.frombuffer(buf, dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio**2) + 1e-9))
        # smooth envelope
        self.smooth = self.smooth * 0.9 + rms * 0.1
        return self.smooth

    def close(self):
        if self.closed:
            return
        self.closed = True

        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=0.5)
        except Exception:
            pass

        try:
            if self.proc and self.proc.stdout:
                self.proc.stdout.close()
        except Exception:
            pass

# ============================================================
# Smooth noise drift (Perlin-like)
# ============================================================

class SmoothNoise:
    def __init__(self, rng):
        self.phase = rng.random(3) * 1000

    def step(self, dt, speed):
        self.phase += dt * speed
        return np.array([
            math.sin(self.phase[0]*0.7),
            math.sin(self.phase[1]*0.9),
            math.sin(self.phase[2]*0.5),
        ])

# ============================================================
# Musical camera
# ============================================================

class CinematicCam:
    def __init__(self, rng, args, fps, frame_w, frame_h):
        self.rng = rng
        self.args = args
        self.fps = fps
        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)
        self.pan_limit = float(np.clip(args.pan_max, 0.0, 1.0))
        self.edge_margin = float(np.clip(args.edge_margin, 0.0, 0.45))

        # state = [cam_center_offset_x_px, cam_center_offset_y_px, zoom, rotation_rad]
        self.state = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        self.vel = np.zeros(4, dtype=np.float64)

        # keep zoom above a floor so pan has room to travel.
        self.zoom_floor = min(
            self.args.zoom_max,
            max(self.args.zoom_floor_min, self.args.zoom_max * self.args.zoom_floor_ratio),
        )
        self.zoom_floor = max(1.0, self.zoom_floor)

        # zoom target changes occasionally to keep motion musical.
        self.target_zoom = self.zoom_floor

        self.next_retarget = 0

        # start with a gentle random heading for straight-line glide.
        theta = rng.uniform(0, math.tau)
        self.pan_speed_px = max(
            self.args.pan_speed_min,
            min(self.frame_w, self.frame_h) * self.args.speed * self.args.pan_speed_mult,
        )
        self.vel[0] = math.cos(theta) * self.pan_speed_px
        self.vel[1] = math.sin(theta) * self.pan_speed_px

        # start already zoomed so panning is visible immediately
        self.state[2] = self.zoom_floor

        # continuous slow rotation + bounce-induced kick
        self.base_rot_vel = math.radians(self.args.rotation_speed) * (1 if rng.random() > 0.5 else -1)
        self.rot_vel = self.base_rot_vel
        self.rot_bounce_target = 0.0
        self.rot_bounce_applied = 0.0
        self.last_bounce = "-"

    def _pan_bounds(self):
        z = max(1.0, float(self.state[2]))

        # Rotation-aware source footprint of the output viewport.
        # This makes bounce logic match visible corners when rotating.
        c = abs(math.cos(float(self.state[3])))
        s = abs(math.sin(float(self.state[3])))
        half_w_src = (c * self.frame_w * 0.5 + s * self.frame_h * 0.5) / z
        half_h_src = (s * self.frame_w * 0.5 + c * self.frame_h * 0.5) / z

        max_tx = max(0.0, (self.frame_w * 0.5 - half_w_src) * self.pan_limit)
        max_ty = max(0.0, (self.frame_h * 0.5 - half_h_src) * self.pan_limit)

        # Optional safety cushion so bounce occurs "near" edge, not exactly on edge.
        cushion = max(0.0, 1.0 - self.edge_margin)
        max_tx = max(0.0, max_tx * cushion)
        max_ty = max(0.0, max_ty * cushion)
        return max_tx, max_ty

    def _retarget(self):
        self.target_zoom = self.rng.uniform(self.zoom_floor, self.args.zoom_max)

    @staticmethod
    def _reflect_1d(pos, vel, bound, dt):
        """Reflect 1D motion off [-bound, +bound] with overshoot handling."""
        if bound <= 1e-9:
            return 0.0, vel, False

        x = pos + vel * dt
        bounced = False

        # Handle rare large-step overshoot by reflecting until inside bounds.
        # (Typically 0 or 1 iterations at video frame rates.)
        for _ in range(4):
            if x > bound:
                x = 2.0 * bound - x
                vel = -abs(vel)
                bounced = True
                continue
            if x < -bound:
                x = -2.0 * bound - x
                vel = abs(vel)
                bounced = True
                continue
            break

        x = float(np.clip(x, -bound, bound))
        return x, vel, bounced

    @staticmethod
    def _advance_billiard_2d(x, y, vx, vy, max_tx, max_ty, dt):
        """Advance one step with exact axis-aligned billiard reflections.

        Uses time-of-impact against vertical/horizontal walls so reflection angle
        matches incidence angle (specular reflection).
        """
        if max_tx <= 1e-12:
            x = 0.0
            vx = abs(vx)
            vx = -vx
        if max_ty <= 1e-12:
            y = 0.0
            vy = abs(vy)
            vy = -vy

        rem = float(dt)
        bounced_x = False
        bounced_y = False

        for _ in range(6):
            if rem <= 1e-10:
                break

            tx_hit = math.inf
            ty_hit = math.inf

            if abs(vx) > 1e-12 and max_tx > 1e-12:
                tx_hit = ((max_tx - x) / vx) if vx > 0 else ((-max_tx - x) / vx)
                if tx_hit < -1e-12:
                    tx_hit = math.inf

            if abs(vy) > 1e-12 and max_ty > 1e-12:
                ty_hit = ((max_ty - y) / vy) if vy > 0 else ((-max_ty - y) / vy)
                if ty_hit < -1e-12:
                    ty_hit = math.inf

            t_hit = min(tx_hit, ty_hit)

            # No collision in remaining interval
            if not np.isfinite(t_hit) or t_hit > rem:
                x += vx * rem
                y += vy * rem
                rem = 0.0
                break

            # Move to impact point
            t_move = max(0.0, t_hit)
            x += vx * t_move
            y += vy * t_move

            hit_x = abs(t_hit - tx_hit) <= 1e-9
            hit_y = abs(t_hit - ty_hit) <= 1e-9

            if hit_x:
                vx = -vx
                bounced_x = True
                x = float(np.clip(x, -max_tx, max_tx))
            if hit_y:
                vy = -vy
                bounced_y = True
                y = float(np.clip(y, -max_ty, max_ty))

            rem -= t_move

        x = float(np.clip(x, -max_tx, max_tx))
        y = float(np.clip(y, -max_ty, max_ty))
        return x, y, vx, vy, bounced_x, bounced_y

    def step(self, dt, frame_idx, audio_level=0.0):
        # periodic zoom retarget
        if frame_idx >= self.next_retarget:
            self._retarget()
            self.next_retarget = frame_idx + int(self.args.retarget_sec*self.fps)

        # zoom first, then compute this frame's pan bounds from original input size.
        zoom_chase = max(0.05, self.args.speed * self.args.zoom_chase)
        self.vel[2] += (self.target_zoom - self.state[2]) * zoom_chase * dt
        self.vel[2] *= self.args.inertia
        self.state[2] += self.vel[2] * dt
        self.state[2] = float(np.clip(self.state[2], self.zoom_floor, self.args.zoom_max))

        # audio zoom react
        if self.args.beat_react:
            self.state[2] *= 1 + audio_level * self.args.beat_strength * 0.05
            self.state[2] = float(np.clip(self.state[2], self.zoom_floor, self.args.zoom_max))

        # maintain nearly constant glide speed (no steering until edge hit)
        speed_now = float(np.linalg.norm(self.vel[:2]))
        if speed_now < 1e-6:
            theta = self.rng.uniform(0, math.tau)
            self.vel[0] = math.cos(theta) * self.pan_speed_px
            self.vel[1] = math.sin(theta) * self.pan_speed_px
        else:
            self.vel[0] *= self.pan_speed_px / speed_now
            self.vel[1] *= self.pan_speed_px / speed_now

        # reflect only when movement crosses current source-frame bounds.
        # Use substeps so bounce appears at/near the visual edge (less jumpy).
        max_tx, max_ty = self._pan_bounds()
        bounced = False
        bounced_x = False
        bounced_y = False

        # keep inside bounds first without changing direction
        self.state[0] = float(np.clip(self.state[0], -max_tx, max_tx))
        self.state[1] = float(np.clip(self.state[1], -max_ty, max_ty))

        old_vx = float(self.vel[0])
        old_vy = float(self.vel[1])

        n_sub = max(1, int(self.args.pan_substeps))
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            self.state[0], self.state[1], self.vel[0], self.vel[1], bx, by = self._advance_billiard_2d(
                float(self.state[0]),
                float(self.state[1]),
                float(self.vel[0]),
                float(self.vel[1]),
                float(max_tx),
                float(max_ty),
                float(dt_sub),
            )
            if bx or by:
                bounced = True
                bounced_x = bounced_x or bx
                bounced_y = bounced_y or by

        if bounced_x and bounced_y:
            bounce_axis = "xy"
        elif bounced_x:
            bounce_axis = "x"
        elif bounced_y:
            bounce_axis = "y"
        else:
            bounce_axis = "-"
        self.last_bounce = bounce_axis

        # continuous slow rotation; bounce can alter angular velocity
        if self.args.rotation_mode == "continuous":
            if bounced:
                if self.args.bounce_style == "physical":
                    # Physical mode: keep billiard translation and set a target
                    # bounce-induced orientation change (applied smoothly below).
                    h0 = math.atan2(old_vy, old_vx)
                    h1 = math.atan2(float(self.vel[1]), float(self.vel[0]))
                    dtheta = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
                    self.rot_bounce_target += dtheta * self.args.bounce_rot_kick
                else:
                    # DVD mode: still specular translation, but add a gentle,
                    # deterministic spin response to wall contact.
                    kick_mag = math.radians(self.args.rotation_speed) * self.args.bounce_rot_kick * 0.30
                    if bounce_axis == "x":
                        # Vertical wall hit: couple spin to vertical travel direction.
                        self.rot_vel += math.copysign(kick_mag, float(self.vel[1]) if abs(self.vel[1]) > 1e-9 else 1.0)
                    elif bounce_axis == "y":
                        # Horizontal wall hit: opposite coupling from horizontal travel.
                        self.rot_vel -= math.copysign(kick_mag, float(self.vel[0]) if abs(self.vel[0]) > 1e-9 else 1.0)
                    else:  # corner hit
                        self.rot_vel = -self.rot_vel

            # Smoothly apply bounce-induced orientation target (physical mode).
            if self.args.bounce_style == "physical":
                e = self.rot_bounce_target - self.rot_bounce_applied
                k = max(0.01, self.args.bounce_rot_ease)
                alpha = 1.0 - math.exp(-k * dt)
                step = e * alpha
                self.rot_bounce_applied += step
                self.state[3] += step

            # ease back to base spin so it's always slow/continuous
            self.rot_vel += (self.base_rot_vel - self.rot_vel) * (1.0 - math.exp(-self.args.rot_return * dt))
            self.state[3] += self.rot_vel * dt
        elif self.args.rotation_mode == "off":
            pass

        return self.state

# ============================================================
# HUD
# ============================================================

def draw_hud(frame, state, audio_level, fps_proc, frame_idx, hud_scale, hud_alpha):
    h,w=frame.shape[:2]
    overlay=frame.copy()

    txt=[
        f"VOIDSTAR CAM",
        f"pan=({state[0]:+.3f},{state[1]:+.3f})",
        f"zoom={state[2]:.3f}",
        f"rot={math.degrees(state[3]):+.2f}",
        f"audio={audio_level:.4f}",
        f"proc_fps={fps_proc:.2f}",
        f"frame={frame_idx}",
    ]

    y=30
    for t in txt:
        cv2.putText(overlay,t,(20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5*hud_scale,(255,255,255),1,cv2.LINE_AA)
        y+=25

    cv2.addWeighted(overlay,hud_alpha,frame,1-hud_alpha,0,frame)

# ============================================================
# Filename builder
# ============================================================

def build_output_name(input_path,args):
    parts=[
        input_path.stem,
        "camwalk",
        f"spd{args.speed:.2f}",
        f"z{args.zoom_max:g}",
        f"dr{args.drift_strength:.2f}",
        f"pm{args.pan_max:.2f}",
        f"em{args.edge_margin:.2f}",
        f"psm{args.pan_speed_mult:.2f}",
        f"psn{args.pan_speed_min:g}",
        f"pss{args.pan_substeps:g}",
        f"zfr{args.zoom_floor_ratio:.2f}",
        f"zfm{args.zoom_floor_min:g}",
        f"zc{args.zoom_chase:.2f}",
        f"rr{args.rot_return:.2f}",
        f"rk{args.bounce_rot_kick:.2f}",
        f"re{args.bounce_rot_ease:.2f}",
        f"bs{args.bounce_style}",
        f"rs{args.rotation_speed:.2f}",
    ]
    if args.beat_react: parts.append("br1")
    if args.bass_react: parts.append("ba1")
    if args.start>0: parts.append(f"s{args.start:g}")
    if args.duration>0: parts.append(f"d{args.duration:g}")
    return "_".join(parts)+".mp4"

# ============================================================
# Main
# ============================================================

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")

    ap.add_argument("--speed",type=float,default=0.18)
    ap.add_argument("--zoom-max",type=float,default=10.0)
    ap.add_argument("--pan-max",type=float,default=1.0)
    ap.add_argument("--edge-margin",type=float,default=0.0,
                    help="Early-bounce margin as fraction of one-sided pan range; 0.0 = true edge")
    ap.add_argument("--pan-speed-mult",type=float,default=0.25,
                    help="Pan speed scale: px/s = min(w,h) * speed * pan_speed_mult")
    ap.add_argument("--pan-speed-min",type=float,default=8.0,
                    help="Minimum pan speed in px/s")
    ap.add_argument("--pan-substeps",type=int,default=4,
                    help="Pan integration substeps per frame (higher = cleaner edge bounces)")
    ap.add_argument("--zoom-floor-ratio",type=float,default=0.22,
                    help="Minimum zoom floor as ratio of zoom-max to allow panning")
    ap.add_argument("--zoom-floor-min",type=float,default=1.25,
                    help="Absolute minimum zoom floor")
    ap.add_argument("--zoom-chase",type=float,default=1.8,
                    help="Zoom chase response rate")
    ap.add_argument("--bounce-rot-kick",type=float,default=0.9,
                    help="Bounce angle to rotation coupling (higher = bigger turn on bounce)")
    ap.add_argument("--bounce-rot-ease",type=float,default=4.0,
                    help="How quickly bounce-induced rotation target is eased in (physical mode)")
    ap.add_argument("--bounce-style",default="dvd",
                    choices=["dvd","physical"],
                    help="dvd = strict screensaver reflections + gentle spin response; physical = billiard reflection + angle-coupled rotation")
    ap.add_argument("--rot-return",type=float,default=2.2,
                    help="How quickly rotation velocity returns to base spin")

    ap.add_argument("--rotation-mode",default="continuous",
                    choices=["continuous","oscillate","off"])
    ap.add_argument("--rotation-speed",type=float,default=1.5)

    ap.add_argument("--retarget-sec",type=float,default=12.0)
    ap.add_argument("--inertia",type=float,default=0.92)
    ap.add_argument("--edge-softness",type=float,default=0.15)

    ap.add_argument("--drift-strength",type=float,default=0.12)
    ap.add_argument("--drift-speed",type=float,default=0.15)
    ap.add_argument("--pan-bias",type=float,default=0.6)

    ap.add_argument("--beat-react",action="store_true")
    ap.add_argument("--beat-strength",type=float,default=0.6)
    ap.add_argument("--bass-react",action="store_true")

    ap.add_argument("--motion-focus",default="none",
                    choices=["none","moving"])

    ap.add_argument("--hud",action="store_true")
    ap.add_argument("--hud-scale",type=float,default=1.0)
    ap.add_argument("--hud-alpha",type=float,default=0.6)
    ap.add_argument("--debug-cam",action="store_true",
                    help="Print script path and camera pan/zoom diagnostics")

    ap.add_argument("--start",type=float,default=0.0)
    ap.add_argument("--duration",type=float,default=0.0)

    ap.add_argument("--codec",default="auto")
    ap.add_argument("--preset",default=None)
    ap.add_argument("--crf",type=int,default=18)

    ap.add_argument("--out",default=None)
    ap.add_argument("--out-dir",default=str(default_videos_dir()))

    args=ap.parse_args()

    if args.debug_cam:
        print(f"[voidstar] script={Path(__file__).resolve()}")
        print(f"[voidstar] cwd={Path.cwd()}")

    input_path=Path(args.input).resolve()
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    if args.out:
        out_path=Path(args.out)
    else:
        out_path=out_dir/build_output_name(input_path,args)

    tmp_video=out_path.with_name(out_path.stem+"__video.mp4")

    cap=cv2.VideoCapture(str(input_path))
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.start>0:
        cap.set(cv2.CAP_PROP_POS_MSEC,args.start*1000)

    if args.duration>0:
        frames_to_process=int(args.duration*fps)
    else:
        frames_to_process=-1

    rng=np.random.default_rng(int(time.time()))
    cam=CinematicCam(rng,args,fps,w,h)

    audio=AudioAnalyzer(input_path,fps,args.start,args.duration) if args.beat_react else None

    codec,preset=pick_codec_and_preset(args.codec,args.preset)

    if codec=="h264_nvenc":
        enc_args=["-c:v","h264_nvenc","-preset",preset,"-cq",str(args.crf)]
    else:
        enc_args=["-c:v","libx264","-preset",preset,"-crf",str(args.crf)]

    ffmpeg_cmd=[
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-f","rawvideo","-pix_fmt","bgr24",
        "-s",f"{w}x{h}","-r",f"{fps}",
        "-i","pipe:0","-an",
        *enc_args,
        "-pix_fmt","yuv420p",
        str(tmp_video)
    ]

    enc=subprocess.Popen(ffmpeg_cmd,stdin=subprocess.PIPE)

    t0=time.time()
    processed=0

    while True:
        if frames_to_process!=-1 and processed>=frames_to_process:
            break

        ok,frame=cap.read()
        if not ok: break

        audio_level=audio.read_level() if audio else 0.0
        x,y,z,r=cam.step(1.0/fps,processed,audio_level)

        tx = float(x)
        ty = float(y)

        # Camera model (dst -> src):
        # src = center + [tx, ty] + R(-r) * (dst-center) / z
        zc = max(1.0, float(z))
        cr = math.cos(float(r))
        sr = math.sin(float(r))
        cx = w * 0.5
        cy = h * 0.5

        a00 = cr / zc
        a01 = sr / zc
        a10 = -sr / zc
        a11 = cr / zc
        m02 = cx + tx - (a00 * cx + a01 * cy)
        m12 = cy + ty - (a10 * cx + a11 * cy)

        M = np.array([[a00, a01, m02],
                  [a10, a11, m12]], dtype=np.float32)

        warped=cv2.warpAffine(frame,M,(w,h),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REFLECT101)

        if args.hud:
            elapsed=time.time()-t0
            fps_proc=processed/max(elapsed,1e-6)
            draw_hud(warped,(x,y,z,r),audio_level,fps_proc,
                     processed,args.hud_scale,args.hud_alpha)

        if args.debug_cam and processed % max(1, int(fps)) == 0:
            # report actual bounds from camera logic (rotation-safe when enabled)
            max_tx, max_ty = cam._pan_bounds()
            z_now = max(1.0, float(z))
            vmag = float(np.linalg.norm(cam.vel[:2]))
            print(
                f"[voidstar][cam] f={processed} pan=({x:+.1f},{y:+.1f}) "
                f"bounds=({max_tx:.1f},{max_ty:.1f}) z={z_now:.2f} "
                f"v={vmag:.1f}px/s bounce={cam.last_bounce}"
            )

        enc.stdin.write(warped.tobytes())
        processed+=1

        if processed%int(fps)==0:
            elapsed=time.time()-t0
            print(f"[voidstar] frames={processed} fps={processed/elapsed:.2f}")

    cap.release()

    if audio:
        audio.close()

    enc.stdin.close()
    enc.wait()

    subprocess.run([
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-i",str(tmp_video),
        "-i",str(input_path),
        "-map","0:v:0","-map","1:a:0?",
        "-c:v","copy","-c:a","copy",
        "-shortest",str(out_path)
    ],check=True)

    tmp_video.unlink(missing_ok=True)

    print(f"Done: {out_path}")
    print(f"Total time: {time.time()-t0:.2f}s")

if __name__=="__main__":
    main()
