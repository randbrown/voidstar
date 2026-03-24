"""
TouchDesigner INCREMENTAL updater for Voidstar network.

Unlike build_voidstar_project.py (full clean rebuild), this script
adds or updates specific elements without destroying manually
customised nodes or parameters.

How to use inside TouchDesigner:
1) Create (or reuse) a Text DAT in /project1 — e.g. "voidstar_updater".
2) Point its File parameter at this script (or paste it in).
3) Toggle the UPDATE_* flags below to choose what to apply.
4) Click Run Script.

Design principles:
- Nodes that already exist are reused, never destroyed.
- Parameters are only overwritten when the update section *explicitly*
  asks for it — your manual tweaks survive by default.
- New nodes are created and wired into the chain at the right place.
- Each update section is an independent function you can enable/disable.
"""

import traceback

# ── LIVE COMP PATH ──────────────────────────────────────────────────
LIVE_PATH = "/project1/voidstar_live"

# ── TOGGLE UPDATE SECTIONS ──────────────────────────────────────────
# Set True for the sections you want to apply on this run.

UPDATE_CAM_ROTATION      = False   # change cam_rotate angle (Flip TOP)
UPDATE_TITLE_TEXT        = False   # change title_hook text content
UPDATE_LOGO_PATH         = False   # change dvdlogo_png file path
ADD_COLOUR_CORRECT       = False   # insert a Colour TOP after level_drive
FIX_OVER_TOPS            = False   # fix logo_over / title_over compositing (switch to Composite TOP)
ADD_AUDIO_REACTIVITY     = False   # bind audio_env to visual parameters (already applied)
ADD_EDGE_OVERLAY         = True    # audio-reactive edge overlay (glitchy flash)
ADD_LOGO_PULSE           = True    # audio-reactive logo scale pulsing
ADD_POSE_OVERLAY         = False   # MediaPipe pose skeleton overlay (Script TOP)
ADD_POINT_TRACK          = False   # optical-flow constellation overlay (Script TOP)
PROBE_PARAMS             = False   # print all parameter names for key nodes (diagnostic)


# ── UPDATE PARAMETERS ───────────────────────────────────────────────
# Only relevant when the matching UPDATE_* flag is True.

# Audio reactivity multipliers (tweak these to taste).
# The envelope from audio_env is typically 0.0 – 0.01 range,
# so large multipliers are needed for visible effect.
AUDIO_GAMMA_MULT    = 300.0     # gamma   = 1.0  + min(2.5,  env * THIS)
AUDIO_DISPLACE_MULT = 1200.0    # weight  = 0.01 + min(0.8,  env * THIS)
AUDIO_BLUR_MULT     = 12000.0   # blur    = 1.0  + min(120,  env * THIS)
AUDIO_NOISE_TX_MULT = 50.0      # noise translate X = env * THIS  (scrolls noise with beats)
AUDIO_NOISE_TY_MULT = 50.0      # noise translate Y = env * THIS

# Edge overlay
AUDIO_EDGE_OPACITY_MULT = 800.0   # edge opacity = min(0.85, env * THIS)

# Logo pulse
LOGO_BASE_SCALE   = 0.24           # must match your transform1 sx/sy
LOGO_PULSE_MULT   = 120.0          # logo scale += min(0.08, env * THIS)

CAM_ROTATION_INDEX = 3          # Flip TOP menu: 0=0°  1=90°  2=180°  3=270°

TITLE_TEXT         = "VOIDSTAR LIVE"
LOGO_FILE          = "../dvd_logo/voidstar_logo_0.png"


# ── HELPERS (same as build script) ──────────────────────────────────

def _safe_set_par(op_obj, par_name, value):
    p = getattr(op_obj.par, par_name, None)
    if p is not None:
        p.val = value


def _safe_set_expr(op_obj, par_name, expr_text):
    p = getattr(op_obj.par, par_name, None)
    if p is not None:
        try:
            p.expr = expr_text
            return True
        except Exception:
            return False
    return False


def _safe_set_expr_any(op_obj, par_names, expr_text):
    for par_name in par_names:
        if _safe_set_expr(op_obj, par_name, expr_text):
            return par_name
    return ""


def _safe_set_par_any(op_obj, par_names, value):
    for par_name in par_names:
        p = getattr(op_obj.par, par_name, None)
        if p is not None:
            p.val = value
            return True
    return False


def _safe_set_menu_any(op_obj, par_names, token_candidates):
    for par_name in par_names:
        p = getattr(op_obj.par, par_name, None)
        if p is None:
            continue
        try:
            names = [str(x).lower() for x in getattr(p, "menuNames", [])]
            labels = [str(x).lower() for x in getattr(p, "menuLabels", [])]
            for token in token_candidates:
                t = str(token).lower()
                for i, n in enumerate(names):
                    if t in n:
                        p.menuIndex = i
                        return par_name
                for i, lbl in enumerate(labels):
                    if t in lbl:
                        p.menuIndex = i
                        return par_name
        except Exception:
            pass
    return ""


def _create_or_get_any(parent, type_names, name):
    existing = parent.op(name)
    if existing is not None:
        return existing
    for n in type_names:
        sym = globals().get(n)
        try:
            if sym is not None:
                return parent.create(sym, name)
        except Exception:
            pass
        try:
            return parent.create(n, name)
        except Exception:
            pass
    raise RuntimeError("Could not create '" + name + "' from: " + ", ".join(type_names))


def _set_node_pos(op_obj, x, y):
    try:
        op_obj.nodeX = x
        op_obj.nodeY = y
    except Exception:
        pass


def _require(parent, name):
    """Return an existing child or raise so the update section can bail out cleanly."""
    node = parent.op(name)
    if node is None:
        raise RuntimeError("Node '" + name + "' not found in " + parent.path)
    return node


def _insert_after(parent, existing_name, new_type_names, new_name):
    """Insert a new node after an existing one, splicing into the chain.

    Returns (new_node, was_created).
    If the new node already exists, returns it untouched (was_created=False).
    """
    existing = _require(parent, existing_name)
    already = parent.op(new_name)
    if already is not None:
        return already, False

    new_node = _create_or_get_any(parent, new_type_names, new_name)

    # Gather everything currently connected to existing's output.
    downstream = []
    try:
        for conn in list(existing.outputConnectors[0].connections):
            downstream.append(conn)
    except Exception:
        pass

    # Wire new_node after existing.
    new_node.inputConnectors[0].connect(existing)

    # Re-wire downstream nodes: replace their input from existing → new_node.
    for conn in downstream:
        target_op = conn.owner
        idx = conn.index
        target_op.inputConnectors[idx].connect(new_node)

    # Position slightly to the right of existing.
    _set_node_pos(new_node, existing.nodeX + 180, existing.nodeY)

    return new_node, True


# ── UPDATE SECTIONS ─────────────────────────────────────────────────

def _add_audio_reactivity(live):
    """Bind audio_env expressions to visual parameters for audio-reactive visuals.

    Targets known parameter names from the user's TD build.
    Safe to re-run: overwrites expressions but not static values.
    """
    env = "abs(op('audio_env')[0])"
    gamma_expr    = "1.0 + min(2.5, ("  + env + " * " + str(AUDIO_GAMMA_MULT)    + "))"
    displace_expr = "0.01 + min(0.8, (" + env + " * " + str(AUDIO_DISPLACE_MULT) + "))"
    blur_expr     = "1.0 + min(120.0, (" + env + " * " + str(AUDIO_BLUR_MULT)    + "))"
    noise_tx_expr = env + " * " + str(AUDIO_NOISE_TX_MULT)
    noise_ty_expr = env + " * " + str(AUDIO_NOISE_TY_MULT)

    results = {}

    # Gamma on level_drive
    level = live.op("level_drive")
    if level:
        r = _safe_set_expr_any(level,
            ["gamma1", "Gamma1", "gamma", "Gamma"],
            gamma_expr)
        results["gamma"] = r or "NO MATCHING PAR"

    # Displacement weight (user's build uses displaceweightx)
    displace = live.op("displace1")
    if displace:
        r = _safe_set_expr_any(displace,
            ["displaceweightx", "Displaceweightx",
             "displaceweight", "Displaceweight",
             "displaceweight1", "Displaceweight1"],
            displace_expr)
        results["displace"] = r or "NO MATCHING PAR"
        # Also try Y/Z displacement channels
        _safe_set_expr(displace, "displaceweighty", displace_expr)
        _safe_set_expr(displace, "displaceweightz", displace_expr)
        _safe_set_expr(displace, "Displaceweighty", displace_expr)
        _safe_set_expr(displace, "Displaceweightz", displace_expr)

    # Blur filter size
    blur = live.op("blur1")
    if blur:
        r = _safe_set_expr_any(blur,
            ["filtersize", "Filtersize",
             "filterwidth", "Filterwidth",
             "size", "Size"],
            blur_expr)
        results["blur"] = r or "NO MATCHING PAR"

    # Noise translate (makes noise scroll with beats → animated displacement)
    noise = live.op("noise1")
    if noise:
        rx = _safe_set_expr_any(noise,
            ["tx", "Tx", "t1", "T1", "translatex", "Translatex"],
            noise_tx_expr)
        ry = _safe_set_expr_any(noise,
            ["ty", "Ty", "t2", "T2", "translatey", "Translatey"],
            noise_ty_expr)
        results["noise_tx"] = rx or "NO MATCHING PAR"
        results["noise_ty"] = ry or "NO MATCHING PAR"

    return results


def _probe_params(live):
    """Print all parameter names for key visual/audio nodes. Diagnostic only."""
    targets = ["level_drive", "displace1", "blur1", "noise1", "edge1",
               "logo_over", "title_over", "audio_env", "transform1",
               "edge_level", "edge_comp"]
    lines = []
    for name in targets:
        node = live.op(name)
        if node is None:
            lines.append(name + ": NOT FOUND")
            continue
        par_names = [p.name for p in node.pars()]
        lines.append(name + " (" + node.OPType + ") params:")
        lines.append("  " + ", ".join(par_names))
    report = "\n".join(lines)
    print(report)
    return report


def _add_edge_overlay(live):
    """Add an audio-reactive edge overlay that flashes on beats.

    Creates:
    - edge_level (levelTOP): controls edge opacity via audio_env expression
    - edge_comp (compositeTOP): composites edges over the displace chain

    Wiring: displace1 → edge_comp[0], edge1 → edge_level → edge_comp[1]
    edge_comp output replaces displace1 in the downstream chain (logo_over).
    """
    results = []

    edge1 = live.op("edge1")
    if edge1 is None:
        return "edge1 not found"

    displace1 = live.op("displace1")
    if displace1 is None:
        return "displace1 not found"

    # Create edge_level to control edge opacity
    edge_level = live.op("edge_level")
    created_level = False
    if edge_level is None:
        edge_level = _create_or_get_any(live, ["levelTOP"], "edge_level")
        created_level = True
        edge_level.inputConnectors[0].connect(edge1)
        _set_node_pos(edge_level, -140, 110)

    # Bind opacity to audio envelope
    env = "abs(op('audio_env')[0])"
    opacity_expr = "min(0.85, " + env + " * " + str(AUDIO_EDGE_OPACITY_MULT) + ")"
    _safe_set_expr_any(edge_level, ["opacity", "Opacity", "opacity1", "Opacity1"], opacity_expr)
    results.append("edge_level: " + ("created" if created_level else "updated"))

    # Create edge_comp to composite edges over the main chain
    edge_comp = live.op("edge_comp")
    created_comp = False
    if edge_comp is None:
        edge_comp = _create_or_get_any(live, ["compositeTOP"], "edge_comp")
        created_comp = True

        # Splice edge_comp between displace1 and its downstream (logo_over)
        downstream = []
        try:
            for conn in list(displace1.outputConnectors[0].connections):
                # Don't redirect edge_level's own connection
                if conn.owner.name != "edge_level" and conn.owner.name != "edge_comp":
                    downstream.append((conn.owner, conn.index))
        except Exception:
            pass

        edge_comp.inputConnectors[0].connect(displace1)
        edge_comp.inputConnectors[1].connect(edge_level)

        for target_op, idx in downstream:
            try:
                target_op.inputConnectors[idx].connect(edge_comp)
            except Exception:
                pass

        _set_node_pos(edge_comp, 60, 110)

    # Set composite operation to "add" for glitchy bright-edge look
    _safe_set_menu_any(edge_comp, ["Operand", "operand", "Operation", "operation"], ["add"])
    results.append("edge_comp: " + ("created" if created_comp else "exists"))

    return ", ".join(results)


def _add_logo_pulse(live):
    """Bind transform1 scale to audio_env for logo pulsing.

    Adds expressions to sx and sy to pulse around LOGO_BASE_SCALE.
    """
    t1 = live.op("transform1")
    if t1 is None:
        return "transform1 not found"

    env = "abs(op('audio_env')[0])"
    pulse_expr = str(LOGO_BASE_SCALE) + " + min(0.08, " + env + " * " + str(LOGO_PULSE_MULT) + ")"

    rx = _safe_set_expr_any(t1, ["sx", "Sx", "scalex", "Scalex"], pulse_expr)
    ry = _safe_set_expr_any(t1, ["sy", "Sy", "scaley", "Scaley"], pulse_expr)

    return "sx=" + str(rx or "MISS") + " sy=" + str(ry or "MISS") + " base=" + str(LOGO_BASE_SCALE)


def _add_pose_overlay(live):
    """Add a Script TOP for MediaPipe pose skeleton overlay.

    Creates:
    - pose_script (scriptTOP): runs MediaPipe per frame on src input
    - pose_comp (compositeTOP): composites pose overlay on chain

    The Script TOP DAT code is written to a Text DAT (pose_script_code).
    """
    results = []

    # Write the Script TOP callback code to a Text DAT
    code_dat = _create_or_get_any(live, ["textDAT"], "pose_script_code")
    code_dat.text = _pose_script_code()
    _set_node_pos(code_dat, -520, -620)
    results.append("pose_script_code DAT written")

    # Create Script TOP
    pose_script = live.op("pose_script")
    created_script = False
    if pose_script is None:
        pose_script = _create_or_get_any(live, ["scriptTOP"], "pose_script")
        created_script = True
        src = live.op("src")
        if src:
            pose_script.inputConnectors[0].connect(src)
        _set_node_pos(pose_script, -320, -620)

    # Point setup DAT to our code
    _safe_set_par_any(pose_script, ["setupdat", "Setupdat", "callbacks", "Callbacks"], "pose_script_code")
    results.append("pose_script: " + ("created" if created_script else "exists"))

    # Create composite to layer pose over the chain
    # Insert after edge_comp if it exists, else after displace1, else after level_drive
    bg_name = "edge_comp"
    if live.op(bg_name) is None:
        bg_name = "displace1"
    if live.op(bg_name) is None:
        bg_name = "level_drive"

    pose_comp = live.op("pose_comp")
    created_comp = False
    if pose_comp is None:
        pose_comp, created_comp = _insert_after(live, bg_name, ["compositeTOP"], "pose_comp")

    if created_comp:
        pose_comp.inputConnectors[1].connect(pose_script)
        _safe_set_menu_any(pose_comp, ["Operand", "operand"], ["over"])

    results.append("pose_comp: " + ("created" if created_comp else "exists"))
    return ", ".join(results)


def _add_point_track(live):
    """Add a Script TOP for optical-flow constellation point tracking.

    Creates:
    - track_script (scriptTOP): runs cv2 feature tracking per frame
    - track_comp (compositeTOP): composites tracking overlay on chain
    """
    results = []

    code_dat = _create_or_get_any(live, ["textDAT"], "track_script_code")
    code_dat.text = _track_script_code()
    _set_node_pos(code_dat, -520, -780)
    results.append("track_script_code DAT written")

    track_script = live.op("track_script")
    created_script = False
    if track_script is None:
        track_script = _create_or_get_any(live, ["scriptTOP"], "track_script")
        created_script = True
        src = live.op("src")
        if src:
            track_script.inputConnectors[0].connect(src)
        _set_node_pos(track_script, -320, -780)

    _safe_set_par_any(track_script, ["setupdat", "Setupdat", "callbacks", "Callbacks"], "track_script_code")
    results.append("track_script: " + ("created" if created_script else "exists"))

    # Insert after pose_comp if it exists, else after edge_comp, else after displace1
    bg_name = "pose_comp"
    if live.op(bg_name) is None:
        bg_name = "edge_comp"
    if live.op(bg_name) is None:
        bg_name = "displace1"
    if live.op(bg_name) is None:
        bg_name = "level_drive"

    track_comp = live.op("track_comp")
    created_comp = False
    if track_comp is None:
        track_comp, created_comp = _insert_after(live, bg_name, ["compositeTOP"], "track_comp")

    if created_comp:
        track_comp.inputConnectors[1].connect(track_script)
        _safe_set_menu_any(track_comp, ["Operand", "operand"], ["over"])

    results.append("track_comp: " + ("created" if created_comp else "exists"))
    return ", ".join(results)


def _pose_script_code():
    """Return the Python code for the pose detection Script TOP."""
    return '''# Voidstar pose skeleton overlay for Script TOP.
# Requires: mediapipe installed in TD's Python environment.
#   In TD's Textport: import pip; pip.main(["install", "mediapipe"])
import numpy as np
try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    _MP_OK = False
    print("[voidstar] mediapipe not installed - pose overlay disabled")

_pose = None
_draw = None

# Voidstar color scheme: cyan connections, bright dots
_CYAN = (255, 255, 0)     # BGR cyan
_WHITE = (255, 255, 255)
_GHOST = (180, 120, 40)   # dim blue-ish for secondary connections

def setup():
    global _pose, _draw
    if not _MP_OK:
        return
    _pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _draw = mp.solutions.drawing_utils

def cook(scriptOP, device, mem):
    if not _MP_OK or _pose is None:
        # Pass through input unchanged
        inp = scriptOP.inputs[0]
        if inp is not None:
            scriptOP.copyNumpyArray(inp.numpyArray(delayed=True))
        return

    inp = scriptOP.inputs[0]
    if inp is None:
        return
    arr = inp.numpyArray(delayed=True)
    if arr is None:
        return

    # arr is RGBA float32 0-1 from TD; convert to uint8 RGB for MediaPipe
    rgb = (arr[:, :, :3] * 255).astype(np.uint8)
    results = _pose.process(rgb)

    # Create transparent overlay (RGBA)
    overlay = np.zeros_like(arr)

    if results.pose_landmarks:
        h, w = rgb.shape[:2]
        lm = results.pose_landmarks.landmark
        pts = []
        for l in lm:
            if l.visibility > 0.5:
                pts.append((int(l.x * w), int(l.y * h)))
            else:
                pts.append(None)

        # Draw connections (Voidstar style: cyan lines)
        connections = mp.solutions.pose.POSE_CONNECTIONS
        for c in connections:
            p1, p2 = pts[c[0]], pts[c[1]]
            if p1 is not None and p2 is not None:
                # Draw on overlay as cyan with alpha
                import cv2
                cv2.line(overlay, p1, p2, (0.0, 1.0, 1.0, 0.7), 2, cv2.LINE_AA)

        # Draw landmarks as bright dots
        for pt in pts:
            if pt is not None:
                import cv2
                cv2.circle(overlay, pt, 4, (1.0, 1.0, 1.0, 0.9), -1, cv2.LINE_AA)

    scriptOP.copyNumpyArray(overlay)
'''


def _track_script_code():
    """Return the Python code for the point tracking Script TOP."""
    return '''# Voidstar constellation point tracker for Script TOP.
# Uses cv2.goodFeaturesToTrack + Lucas-Kanade optical flow.
import numpy as np
import cv2

_prev_gray = None
_prev_pts = None
_MAX_POINTS = 80
_DETECT_INTERVAL = 5  # re-detect features every N frames
_frame_count = 0
_LINK_RADIUS = 120   # pixels: max distance for constellation links

# Feature detection params
_FEATURE_PARAMS = dict(
    maxCorners=_MAX_POINTS,
    qualityLevel=0.02,
    minDistance=15,
    blockSize=7,
)

# Lucas-Kanade params
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

def setup():
    pass

def cook(scriptOP, device, mem):
    global _prev_gray, _prev_pts, _frame_count

    inp = scriptOP.inputs[0]
    if inp is None:
        return
    arr = inp.numpyArray(delayed=True)
    if arr is None:
        return

    # Convert to uint8 grayscale for tracking
    rgb = (arr[:, :, :3] * 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Create transparent overlay
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    _frame_count += 1

    if _prev_gray is None or _prev_pts is None or _frame_count % _DETECT_INTERVAL == 0:
        new_pts = cv2.goodFeaturesToTrack(gray, **_FEATURE_PARAMS)
        if new_pts is not None:
            _prev_pts = new_pts
        _prev_gray = gray
        if _prev_pts is None:
            scriptOP.copyNumpyArray(overlay)
            return

    # Track points
    cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        _prev_gray, gray, _prev_pts, None, **_LK_PARAMS
    )

    if cur_pts is not None and status is not None:
        good_new = cur_pts[status.ravel() == 1]
        if len(good_new) > 0:
            pts = good_new.reshape(-1, 2)

            # Draw constellation links (connect nearby points)
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dx = pts[i][0] - pts[j][0]
                    dy = pts[i][1] - pts[j][1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < _LINK_RADIUS:
                        alpha = 0.6 * (1.0 - dist / _LINK_RADIUS)
                        p1 = (int(pts[i][0]), int(pts[i][1]))
                        p2 = (int(pts[j][0]), int(pts[j][1]))
                        cv2.line(overlay, p1, p2, (0.0, 1.0, 1.0, alpha), 1, cv2.LINE_AA)

            # Draw points as bright dots
            for pt in pts:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 3, (1.0, 1.0, 1.0, 0.8), -1, cv2.LINE_AA)

            _prev_pts = good_new.reshape(-1, 1, 2)
        else:
            _prev_pts = None
    else:
        _prev_pts = None

    _prev_gray = gray
    scriptOP.copyNumpyArray(overlay)
'''


def _update_cam_rotation(live):
    """Change the Flip TOP rotation index."""
    cam = live.op("cam_rotate")
    if cam is None:
        return "cam_rotate not found"
    _safe_set_menu_any(cam, ["Rotate", "rotate"], [str(CAM_ROTATION_INDEX)])
    _safe_set_par_any(cam, ["Rotate", "rotate"], CAM_ROTATION_INDEX)
    return "set to index " + str(CAM_ROTATION_INDEX)


def _update_title_text(live):
    title = live.op("title_hook")
    if title is None:
        return "title_hook not found"
    _safe_set_par_any(title, ["Text", "text"], TITLE_TEXT)
    return TITLE_TEXT


def _update_logo_path(live):
    logo = live.op("dvdlogo_png")
    if logo is None:
        return "dvdlogo_png not found"
    _safe_set_par_any(logo, ["File", "file"], LOGO_FILE)
    return LOGO_FILE


def _add_colour_correct(live):
    """Insert a Colour Correct TOP after level_drive (before the rest of the chain).

    Only creates it if it doesn't already exist; existing nodes are never moved.
    """
    node, created = _insert_after(live, "level_drive", ["hsvadjustTOP"], "colour_correct")
    if created:
        return "created and wired"
    return "already exists (untouched)"


def _fix_over_tops(live):
    """Replace Over TOPs with Composite TOPs for reliable compositing.

    Destroys old Over TOPs and creates Composite TOPs in their place,
    preserving the same wiring.
    """
    results = []
    replacements = [
        # (name, upstream_input0, upstream_input1)
        ("logo_over", "displace1", "dvdlogo_png"),
        ("title_over", "logo_over", "title_hook"),
    ]
    for name, in0_name, in1_name in replacements:
        old = live.op(name)
        if old is not None:
            # Gather downstream connections before destroying.
            downstream = []
            try:
                for conn in list(old.outputConnectors[0].connections):
                    downstream.append((conn.owner, conn.index))
            except Exception:
                pass
            try:
                old.destroy()
            except Exception:
                results.append(name + ": could not destroy old Over TOP")
                continue

        new_node = _create_or_get_any(live, ["compositeTOP"], name)
        in0 = live.op(in0_name)
        in1 = live.op(in1_name)
        if in0:
            new_node.inputConnectors[0].connect(in0)
        if in1:
            new_node.inputConnectors[1].connect(in1)
        # Re-wire downstream.
        if old is not None:
            for target_op, idx in downstream:
                try:
                    target_op.inputConnectors[idx].connect(new_node)
                except Exception:
                    pass
        _safe_set_menu_any(new_node, ["Operand", "operand", "Operation", "operation"], ["over"])
        _safe_set_par_any(new_node, ["Operand", "operand", "Operation", "operation"], 0)
        results.append(name + ": replaced with Composite TOP")
    return ", ".join(results)


# ── MAIN ────────────────────────────────────────────────────────────

def update():
    live = op(LIVE_PATH)
    if live is None:
        msg = (
            "Cannot find " + LIVE_PATH + ".\n"
            "Run build_voidstar_project.py first to create the initial network."
        )
        print(msg)
        try:
            ui.messageBox("Voidstar Updater", msg)
        except Exception:
            pass
        return

    log_lines = ["Voidstar updater ran."]
    any_enabled = False

    try:
        if UPDATE_CAM_ROTATION:
            any_enabled = True
            r = _update_cam_rotation(live)
            log_lines.append("Cam rotation: " + str(r))

        if UPDATE_TITLE_TEXT:
            any_enabled = True
            r = _update_title_text(live)
            log_lines.append("Title text: " + str(r))

        if UPDATE_LOGO_PATH:
            any_enabled = True
            r = _update_logo_path(live)
            log_lines.append("Logo path: " + str(r))

        if ADD_COLOUR_CORRECT:
            any_enabled = True
            r = _add_colour_correct(live)
            log_lines.append("Colour correct: " + str(r))

        if FIX_OVER_TOPS:
            any_enabled = True
            r = _fix_over_tops(live)
            log_lines.append("Over TOPs: " + str(r))

        if ADD_AUDIO_REACTIVITY:
            any_enabled = True
            r = _add_audio_reactivity(live)
            log_lines.append("Audio reactivity: " + str(r))

        if ADD_EDGE_OVERLAY:
            any_enabled = True
            r = _add_edge_overlay(live)
            log_lines.append("Edge overlay: " + str(r))

        if ADD_LOGO_PULSE:
            any_enabled = True
            r = _add_logo_pulse(live)
            log_lines.append("Logo pulse: " + str(r))

        if ADD_POSE_OVERLAY:
            any_enabled = True
            r = _add_pose_overlay(live)
            log_lines.append("Pose overlay: " + str(r))

        if ADD_POINT_TRACK:
            any_enabled = True
            r = _add_point_track(live)
            log_lines.append("Point tracking: " + str(r))

        if PROBE_PARAMS:
            any_enabled = True
            r = _probe_params(live)
            log_lines.append("Param probe:\n" + r)

        if not any_enabled:
            log_lines.append("No update sections enabled. Set UPDATE_* flags to True.")

    except Exception:
        err = traceback.format_exc()
        log_lines.append("ERROR:\n" + err)
        print(err)

    summary = "\n".join(log_lines)
    print(summary)

    # Write log to a DAT so it persists.
    project = op("/project1")
    if project:
        log_dat = _create_or_get_any(project, ["textDAT"], "voidstar_update_log")
        log_dat.text = summary
        log_dat.nodeX = 120
        log_dat.nodeY = -220

    try:
        ui.messageBox("Voidstar Updater", summary)
    except Exception:
        pass


update()
