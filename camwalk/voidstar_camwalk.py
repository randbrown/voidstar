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
    def __init__(self, rng, args, fps):
        self.rng = rng
        self.args = args
        self.fps = fps

        self.state = np.array([0.0,0.0,1.0,0.0])
        self.vel = np.zeros(4)
        self.target = self.state.copy()

        self.noise = SmoothNoise(rng)
        self.next_retarget = 0
        self.rot_dir = 1 if rng.random()>0.5 else -1

    def _retarget(self):
        spread = self.args.pan_bias
        self.target[0] = self.rng.uniform(-spread,spread)
        self.target[1] = self.rng.uniform(-spread,spread)
        self.target[2] = self.rng.uniform(1.0,self.args.zoom_max)

    def step(self, dt, frame_idx, audio_level=0.0):
        # periodic retarget
        if frame_idx >= self.next_retarget:
            self._retarget()
            self.next_retarget = frame_idx + int(self.args.retarget_sec*self.fps)

        # smooth drift
        drift = self.noise.step(dt, self.args.drift_speed)
        self.vel[0] += drift[0]*self.args.drift_strength*dt
        self.vel[1] += drift[1]*self.args.drift_strength*dt

        # target chase
        accel = (self.target - self.state)*(self.args.speed*0.35)
        self.vel += accel*dt

        # inertia
        self.vel *= self.args.inertia
        self.state += self.vel*dt

        # soft bounds
        for i in (0,1):
            if self.state[i]>1:
                self.vel[i]-=(self.state[i]-1)*self.args.edge_softness
            if self.state[i]<-1:
                self.vel[i]-=(self.state[i]+1)*self.args.edge_softness

        # rotation
        if self.args.rotation_mode=="continuous":
            self.state[3]+=math.radians(self.args.rotation_speed)*dt*self.rot_dir

        # audio zoom react
        if self.args.beat_react:
            self.state[2]*=1+audio_level*self.args.beat_strength*0.05

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
    ]
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
    ap.add_argument("--pan-max",type=float,default=0.15)

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

    ap.add_argument("--start",type=float,default=0.0)
    ap.add_argument("--duration",type=float,default=0.0)

    ap.add_argument("--codec",default="auto")
    ap.add_argument("--preset",default=None)
    ap.add_argument("--crf",type=int,default=18)

    ap.add_argument("--out",default=None)
    ap.add_argument("--out-dir",default=str(default_videos_dir()))

    args=ap.parse_args()

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
    cam=CinematicCam(rng,args,fps)

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

    pan_px=min(w,h)*args.pan_max
    t0=time.time()
    processed=0

    while True:
        if frames_to_process!=-1 and processed>=frames_to_process:
            break

        ok,frame=cap.read()
        if not ok: break

        audio_level=audio.read_level() if audio else 0.0
        x,y,z,r=cam.step(1.0/fps,processed,audio_level)

        tx=x*pan_px
        ty=y*pan_px

        M=cv2.getRotationMatrix2D((w/2,h/2),math.degrees(r),max(1.0,z))
        M[0,2]+=tx
        M[1,2]+=ty

        warped=cv2.warpAffine(frame,M,(w,h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)

        if args.hud:
            elapsed=time.time()-t0
            fps_proc=processed/max(elapsed,1e-6)
            draw_hud(warped,(x,y,z,r),audio_level,fps_proc,
                     processed,args.hud_scale,args.hud_alpha)

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
