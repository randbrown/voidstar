"""
TouchDesigner project creator — Voidstar Black Hole Simulation.

Audio-reactive 3D black hole with:
  - Event horizon   : pulsing black sphere (radius driven by audio)
  - Accretion disc  : torus SOP with animated noise displacement + emissive material
  - Infalling matter: circle SOP ring with spiral noise — rendered as point particles
  - Camera + lights + renderTOP 3D pipeline
  - Feedback trail accumulation (slower decay on louder audio = more vivid trails)
  - Gravitational lensing warp (Displace TOP driven by animated noise)
  - Bloom / glow (Blur TOP — radius driven by audio)
  - Audio input chain: live (loopback + instrument) OR audio file, switchable at runtime

Usage inside TouchDesigner
--------------------------
RECOMMENDED — link the DAT to the file (avoids stale-paste bugs):
  1. Create a Text DAT in /project1.
  2. In its parameters, set the File field to the full path of this .py file
     (e.g. C:/Users/.../voidstar_td_starter/build_blackhole_project.py).
  3. Click the ↑ Load arrow next to the File field to pull content from disk.
  4. Right-click the DAT → Run Script.
  5. Re-click Load + Run Script whenever you want to pick up changes.

ALTERNATIVE — plain paste:
  1. Create a Text DAT in /project1.
  2. Open the editor, select all (Ctrl+A), delete, paste the file content.
  3. Right-click the DAT → Run Script.
  NOTE: If you keep getting errors from old line numbers, the DAT still has
  stale content — repeat step 2.

Audio switch
------------
  audio_src_switch  Index = 0  → live inputs  (loopback + instrument merged)
  audio_src_switch  Index = 1  → audio_file   (set the File param on that CHOP)

Audio interfaces
----------------
  audio_loopback   : software / system audio bus (default 2-channel)
  audio_instrument : external interface / instrument input

Kinect
------
  Not wired by default.  To add hand-position control, append a kinectCHOP and
  route its x/y channels into cam.Tx / cam.Ty expressions.

Clean rebuild
-------------
  Set CLEAN_REBUILD = True (below) to destroy and recreate the network on each run.
  Set to False to preserve any manual edits you added inside blackhole_live.
"""

import traceback

CLEAN_REBUILD = True


# ─── helpers (shared pattern with build_voidstar_project.py) ──────────────────

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
            names  = [str(x).lower() for x in getattr(p, "menuNames",  [])]
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
        # 1. try a global symbol (user-imported or TD-injected)
        sym = globals().get(n)
        # 2. fall back to td.<n> — the reliable way to get TD operator classes
        if sym is None:
            try:
                sym = getattr(td, n, None)
            except Exception:
                pass
        try:
            if sym is not None:
                return parent.create(sym, name)
        except Exception:
            pass
        try:
            return parent.create(n, name)
        except Exception:
            pass
    raise RuntimeError(
        "Could not create operator '{}' from types: {}".format(
            name, ", ".join(type_names)
        )
    )


def _set_node_pos(op_obj, x, y):
    try:
        op_obj.nodeX = x
        op_obj.nodeY = y
    except Exception:
        pass


def _set_viewer(op_obj, enabled):
    try:
        op_obj.viewer = enabled
    except Exception:
        pass


def _layout_children(parent):
    try:
        parent.layoutChildren(horizontal=True, vertical=True)
    except Exception:
        pass


def _disconnect_all_inputs(op_obj):
    try:
        for conn in op_obj.inputConnectors:
            for c in list(conn.connections):
                c.disconnect()
    except Exception:
        pass


def _destroy_if_exists(parent, name):
    op_obj = parent.op(name)
    if op_obj is None:
        return False
    try:
        op_obj.destroy()
        return True
    except Exception:
        return False


def _write_build_log(project, message):
    log = _create_or_get_any(project, ["textDAT"], "blackhole_build_log")
    log.text = message
    _set_node_pos(log, 0, -240)
    return log


# ─── audio diagnostic DAT ────────────────────────────────────────────────────

def _audio_diag_text():
    return (
        "\"\"\"Audio diagnostics for /project1/blackhole_live.\"\"\"\n"
        "\n"
        "def _absmax(chop_op):\n"
        "    if chop_op is None:\n"
        "        return 0.0\n"
        "    vals = []\n"
        "    try:\n"
        "        for ch in chop_op.chans():\n"
        "            try:\n"
        "                vals.append(abs(ch.eval()))\n"
        "            except Exception:\n"
        "                pass\n"
        "    except Exception:\n"
        "        pass\n"
        "    return max(vals) if vals else 0.0\n"
        "\n"
        "def report(path='/project1/blackhole_live', threshold=0.001):\n"
        "    parent = op(path)\n"
        "    if parent is None:\n"
        "        print('[RED] Missing ' + path)\n"
        "        return\n"
        "    inst   = parent.op('audio_instrument')\n"
        "    loop   = parent.op('audio_loopback')\n"
        "    afile  = parent.op('audio_file')\n"
        "    env    = parent.op('audio_env')\n"
        "    sw     = parent.op('audio_src_switch')\n"
        "    src    = 'live' if (sw and sw.par.Index.val == 0) else 'file'\n"
        "    v_inst = _absmax(inst)\n"
        "    v_loop = _absmax(loop)\n"
        "    v_file = _absmax(afile)\n"
        "    v_env  = _absmax(env)\n"
        "    ok = lambda v: '[GREEN]' if v > threshold else '[RED]  '\n"
        "    print('Black hole audio diagnostics  (active source: ' + src + ')')\n"
        "    print(ok(v_inst) + ' audio_instrument  max=' + ('%.6f' % v_inst))\n"
        "    print(ok(v_loop) + ' audio_loopback    max=' + ('%.6f' % v_loop))\n"
        "    print(ok(v_file) + ' audio_file        max=' + ('%.6f' % v_file))\n"
        "    print(ok(v_env)  + ' audio_env (out)   max=' + ('%.6f' % v_env))\n"
        "    print(('[GREEN] live OK' if (v_inst > threshold or v_loop > threshold) else '[RED]  live silent'))\n"
        "\n"
        "report()\n"
    )


def _try_create_mat(parent, name, rgb=(1.0, 1.0, 1.0)):
    """Create a Constant MAT in *parent* with the given RGB colour.

    Returns the MAT op on success, or None on failure (build continues).
    Tries every known TD type string for the Constant material.
    """
    mat_types = ["constantMAT", "constMAT", "phongMAT"]
    mat = None
    existing = parent.op(name)
    if existing is not None:
        mat = existing
    else:
        for t in mat_types:
            try:
                mat = parent.create(t, name)
                break
            except Exception:
                pass
    if mat is None:
        return None
    # Set colour — try every known parameter name variant
    r, g, b = rgb
    for par in ["Colorr", "Cr", "Difcolorr", "Color1r", "colorr"]:
        _safe_set_par(mat, par, r)
    for par in ["Colorg", "Cg", "Difcolorg", "Color1g", "colorg"]:
        _safe_set_par(mat, par, g)
    for par in ["Colorb", "Cb", "Difcolorb", "Color1b", "colorb"]:
        _safe_set_par(mat, par, b)
    for par in ["Colora", "Ca", "Difcolora", "Color1a", "colora"]:
        _safe_set_par(mat, par, 1.0)
    return mat


def _assign_mat_to_geo(geo_comp, mat):
    """Point a geometryCOMP at a MAT via its Render-page Material parameter.

    This is more reliable than wiring a materialSOP inside the geo COMP because
    it doesn't require creating additional operators or knowing the internal
    materialSOP type string.
    """
    if mat is None:
        return
    for par in ["Material", "material", "Mat", "mat"]:
        p = getattr(geo_comp.par, par, None)
        if p is not None:
            try:
                p.val = mat.path
                return
            except Exception:
                pass



def _build_geo_event_horizon(parent):
    """Black sphere at the origin — the event horizon."""
    geo    = _create_or_get_any(parent, ["geometryCOMP"], "geo_event_horizon")
    sphere = _create_or_get_any(geo, ["sphereSOP", "spherePOP", "sphere"], "sphere1")
    out    = _create_or_get_any(geo, ["nullSOP", "null"],   "out1")

    _safe_set_par_any(sphere, ["Rows", "rows"], 56)
    _safe_set_par_any(sphere, ["Cols", "cols"], 56)

    _disconnect_all_inputs(out)
    try:
        out.inputConnectors[0].connect(sphere)
    except Exception:
        pass  # viewer will still show the sphere if wiring fails

    mat = _try_create_mat(parent, "mat_horizon", rgb=(0.0, 0.0, 0.0))
    _assign_mat_to_geo(geo, mat)

    a = "abs(op('../audio_env')[0])"
    radius_expr = "1.4 + min(0.7, {a} * 8.0)".format(a=a)
    for par in ["Radx", "Radius", "Rad", "rad"]:
        _safe_set_expr(sphere, par, radius_expr)
    for par in ["Rady", "Radiusy", "Rady1"]:
        _safe_set_expr(sphere, par, radius_expr)
    for par in ["Radz", "Radiusz", "Radz1"]:
        _safe_set_expr(sphere, par, radius_expr)

    _set_node_pos(sphere, -200, 0)
    _set_node_pos(out,     200, 0)
    _set_viewer(out, True)
    return geo


# ─── geometry COMP: accretion disc ───────────────────────────────────────────

def _build_geo_accretion_disc(parent):
    """Torus SOP with animated noise displacement.

    In TD 2025 the torus operator is a POP (torusPOP), which belongs to a
    different operator family from noiseSOP/nullSOP and cannot be wired to
    them.  When that happens we fall back to a thick circleSOP ring which IS
    a SOP and can be displaced by noiseSOP.
    """
    geo   = _create_or_get_any(parent, ["geometryCOMP"], "geo_disc")
    noise = _create_or_get_any(geo, ["noiseSOP", "noise"], "noise_disc")
    out   = _create_or_get_any(geo, ["nullSOP",  "null"],  "out1")

    # Try torusSOP first; if TD hands us a torusPOP destroy it and use circle.
    torus_raw = None
    for tname in ["torusSOP", "torusPOP", "torus"]:
        try:
            torus_raw = _create_or_get_any(geo, [tname], "torus1")
            break
        except Exception:
            pass

    # Determine whether the created op is a SOP (wirable to noiseSOP/nullSOP).
    # Any POP class name ends in 'POP'; SOP class names end in 'SOP'.
    is_sop = False
    if torus_raw is not None:
        cls_name = type(torus_raw).__name__
        if cls_name.endswith("SOP"):
            is_sop = True
        elif cls_name.endswith("POP"):
            is_sop = False  # POP — destroy and use circle fallback
        else:
            # Try a test connect to determine family
            try:
                noise.inputConnectors[0].connect(torus_raw)
                _disconnect_all_inputs(noise)
                is_sop = True
            except Exception:
                is_sop = False

    if not is_sop:
        # Destroy the unusable POP; replace with a circleSOP disc ring.
        if torus_raw is not None:
            try:
                torus_raw.destroy()
            except Exception:
                pass
        disc_sop = _create_or_get_any(geo, ["circleSOP", "circle"], "disc1")
        _safe_set_par_any(disc_sop, ["Radius", "Rad", "Radx", "rad"],  2.6)
        _safe_set_par_any(disc_sop, ["Divs",   "Divisions", "divs"],   120)
        # XZ plane so it lies flat
        _safe_set_menu_any(disc_sop,
                           ["Orient", "orient", "Orientation", "orientation"],
                           ["xz", "zx", "0"])
        shape_op = disc_sop
    else:
        shape_op = torus_raw
        _safe_set_par_any(shape_op, ["Radx", "Radius",  "Rad",  "rad"],  2.6)
        _safe_set_par_any(shape_op, ["Rady", "Radiusy", "Rad2", "rad2"], 0.38)
        _safe_set_par_any(shape_op, ["Rows", "Divsu",   "rows"], 120)
        _safe_set_par_any(shape_op, ["Cols", "Divsv",   "cols"],  44)

    _safe_set_par_any(noise, ["Period", "period", "Freq", "freq"], 0.65)
    _safe_set_expr_any(noise, ["Tx", "tx", "Offsetx", "offsetx"],
                       "absTime.seconds * 0.9")
    _safe_set_expr_any(noise, ["Ty", "ty", "Offsety", "offsety"],
                       "absTime.seconds * 0.35")
    _safe_set_expr_any(noise, ["Tz", "tz", "Offsetz", "offsetz"],
                       "absTime.seconds * -0.55")

    a = "abs(op('../audio_env')[0])"
    _safe_set_expr_any(noise, ["Amp", "Amplitude", "amp", "Gain", "gain"],
                       "0.08 + min(0.9, {a} * 14.0)".format(a=a))

    _disconnect_all_inputs(noise)
    _disconnect_all_inputs(out)
    noise.inputConnectors[0].connect(shape_op)
    out.inputConnectors[0].connect(noise)

    mat = _try_create_mat(parent, "mat_disc", rgb=(0.92, 0.28, 0.04))
    if mat is not None:
        _safe_set_expr_any(mat, ["Colorr", "Cr", "Difcolorr", "Color1r"],
                           "min(1.0, 0.92 + {a} * 0.5)".format(a=a))
        _safe_set_expr_any(mat, ["Colorg", "Cg", "Difcolorg", "Color1g"],
                           "min(1.0, 0.28 + {a} * 8.0)".format(a=a))
        _safe_set_expr_any(mat, ["Colorb", "Cb", "Difcolorb", "Color1b"],
                           "min(1.0, 0.04 + {a} * 4.0)".format(a=a))
    _assign_mat_to_geo(geo, mat)

    _set_node_pos(shape_op, -320, 0)
    _set_node_pos(noise,    -120, 0)
    _set_node_pos(out,       280, 0)
    _set_viewer(out, True)
    return geo



# ─── geometry COMP: infalling particle ring ───────────────────────────────────

def _build_geo_particles(parent):
    """Ring of infalling matter: circle SOP → noise SOP → point render."""
    geo    = _create_or_get_any(parent, ["geometryCOMP"], "geo_particles")
    circle = _create_or_get_any(geo, ["circleSOP", "circlePOP", "circle"], "circle1")
    noise  = _create_or_get_any(geo, ["noiseSOP", "noise"],  "noise_spiral")
    out    = _create_or_get_any(geo, ["nullSOP", "null"],   "out1")

    _safe_set_par_any(circle, ["Radius", "Rad", "Radx",     "rad"],  3.5)
    _safe_set_par_any(circle, ["Divs",   "Divisions", "divs"],       240)
    _safe_set_menu_any(circle,
                       ["Orient", "orient", "Orientation", "orientation"],
                       ["xz", "zx", "0"])

    _safe_set_par_any(noise, ["Period", "period", "Freq", "freq"], 0.9)
    _safe_set_expr_any(noise, ["Tx", "tx", "Offsetx", "offsetx"],
                       "absTime.seconds * -1.4")
    _safe_set_expr_any(noise, ["Ty", "ty", "Offsety", "offsety"],
                       "absTime.seconds * 0.4")
    _safe_set_expr_any(noise, ["Tz", "tz", "Offsetz", "offsetz"],
                       "absTime.seconds * 0.85")

    a = "abs(op('../audio_env')[0])"
    _safe_set_expr_any(circle, ["Radius", "Rad", "Radx", "rad"],
                       "3.2 + min(2.0, {a} * 10.0)".format(a=a))
    _safe_set_expr_any(noise, ["Amp", "Amplitude", "amp", "Gain", "gain"],
                       "0.22 + min(1.2, {a} * 12.0)".format(a=a))

    _disconnect_all_inputs(noise)
    _disconnect_all_inputs(out)
    try:
        noise.inputConnectors[0].connect(circle)
        out.inputConnectors[0].connect(noise)
    except Exception:
        _disconnect_all_inputs(out)
        out.inputConnectors[0].connect(circle)

    mat = _try_create_mat(parent, "mat_particles", rgb=(1.0, 0.88, 0.55))
    _assign_mat_to_geo(geo, mat)

    _safe_set_par_any(geo, ["Pspritesize", "Pspritesizex", "Pointsize",
                             "pspritesize", "pointsize"], 6.0)

    _set_node_pos(circle, -420, 0)
    _set_node_pos(noise,  -220, 0)
    _set_node_pos(out,     180, 0)
    _set_viewer(out, True)
    return geo


# ─── full blackhole_live network ─────────────────────────────────────────────

def _build_blackhole(parent_comp):
    base = _create_or_get_any(parent_comp, ["baseCOMP"], "blackhole_live")
    base.nodeX = -300
    base.nodeY =  100

    section_errors = []

    # ── AUDIO CHAIN ──────────────────────────────────────────────────────────
    #
    # Live path:  loopback + instrument → live_merge → ─┐
    #                                                     ├→ src_switch → analyze → math → filter → audio_env
    # File path:  audio_file ──────────────────────────→ ─┘
    #
    audio_env = None
    try:
        audio_inst        = _create_or_get_any(base, ["audiodeviceinCHOP", "audiodevinCHOP"], "audio_instrument")
        audio_loop        = _create_or_get_any(base, ["audiodeviceinCHOP", "audiodevinCHOP"], "audio_loopback")
        audio_file        = _create_or_get_any(base, ["audiofileinCHOP"],                     "audio_file")
        audio_live_merge  = _create_or_get_any(base, ["mergeCHOP"],                           "audio_live_merge")
        audio_src_switch  = _create_or_get_any(base, ["switchCHOP"],                          "audio_src_switch")
        audio_analyze     = _create_or_get_any(base, ["analyzeCHOP"],                         "audio_analyze")
        audio_math        = _create_or_get_any(base, ["mathCHOP"],                            "audio_math")
        audio_filter      = _create_or_get_any(base, ["filterCHOP"],                          "audio_filter")
        audio_env         = _create_or_get_any(base, ["nullCHOP"],                            "audio_env")
        audio_diag        = _create_or_get_any(base, ["textDAT"],                             "audio_diag")

        _disconnect_all_inputs(audio_live_merge)
        _disconnect_all_inputs(audio_src_switch)
        _disconnect_all_inputs(audio_analyze)
        _disconnect_all_inputs(audio_math)
        _disconnect_all_inputs(audio_filter)
        _disconnect_all_inputs(audio_env)

        audio_live_merge.inputConnectors[0].connect(audio_loop)
        audio_live_merge.inputConnectors[1].connect(audio_inst)
        audio_src_switch.inputConnectors[0].connect(audio_live_merge)
        audio_src_switch.inputConnectors[1].connect(audio_file)
        audio_analyze.inputConnectors[0].connect(audio_src_switch)
        audio_math.inputConnectors[0].connect(audio_analyze)
        audio_filter.inputConnectors[0].connect(audio_math)
        audio_env.inputConnectors[0].connect(audio_filter)

        _safe_set_menu_any(audio_analyze, ["Function", "function"], ["rms", "root mean"])
        _safe_set_par(audio_analyze,   "Function",    "rms")
        _safe_set_par(audio_filter,    "Filterwidth", 0.06)
        _safe_set_par(audio_env,       "Cooktype",    "always")
        _safe_set_par_any(audio_loop,        ["Numchans", "numchans", "Chans", "chans"], 2)
        _safe_set_par_any(audio_src_switch,  ["Index", "index"], 0)
        audio_diag.text = _audio_diag_text()
    except Exception:
        section_errors.append("AUDIO:\n" + traceback.format_exc())

    # ── 3D SCENE ─────────────────────────────────────────────────────────────
    cam = None
    try:
        cam    = _create_or_get_any(base, ["cameraCOMP"], "cam")
        light1 = _create_or_get_any(base, ["lightCOMP"],  "light_key")
        light2 = _create_or_get_any(base, ["lightCOMP"],  "light_fill")

        _safe_set_par_any(cam, ["Tx", "tx"],   0.0)
        _safe_set_par_any(cam, ["Ty", "ty"],   7.0)
        _safe_set_par_any(cam, ["Tz", "tz"],  14.0)
        _safe_set_par_any(cam, ["Rx", "rx"], -26.0)
        _safe_set_par_any(cam, ["Ry", "ry"],   0.0)
        _safe_set_par_any(cam, ["Rz", "rz"],   0.0)
        _safe_set_par_any(cam, ["Fov",  "fov",  "Fovx", "fovx"],  55.0)
        _safe_set_par_any(cam, ["Near", "near", "Clip", "clipn"], 0.01)
        _safe_set_par_any(cam, ["Far",  "far",  "Clipf"],         500.0)

        _safe_set_par_any(light1, ["Tx", "tx"],   6.0)
        _safe_set_par_any(light1, ["Ty", "ty"],   9.0)
        _safe_set_par_any(light1, ["Tz", "tz"],   4.0)
        _safe_set_par_any(light1, ["Colorr", "Cr", "Lightr"], 1.00)
        _safe_set_par_any(light1, ["Colorg", "Cg", "Lightg"], 0.88)
        _safe_set_par_any(light1, ["Colorb", "Cb", "Lightb"], 0.62)
        _safe_set_par_any(light1, ["Intensity", "intensity", "Dimmer", "dimmer"], 0.90)

        _safe_set_par_any(light2, ["Tx", "tx"],  -5.0)
        _safe_set_par_any(light2, ["Ty", "ty"],   3.0)
        _safe_set_par_any(light2, ["Tz", "tz"],  -4.0)
        _safe_set_par_any(light2, ["Colorr", "Cr", "Lightr"], 0.18)
        _safe_set_par_any(light2, ["Colorg", "Cg", "Lightg"], 0.22)
        _safe_set_par_any(light2, ["Colorb", "Cb", "Lightb"], 0.52)
        _safe_set_par_any(light2, ["Intensity", "intensity", "Dimmer", "dimmer"], 0.30)
    except Exception:
        section_errors.append("CAM/LIGHTS:\n" + traceback.format_exc())

    # Each geo COMP is isolated — a SOP/MAT failure in one won't block the others.
    try:
        _build_geo_event_horizon(base)
    except Exception:
        section_errors.append("GEO_HORIZON:\n" + traceback.format_exc())

    try:
        _build_geo_accretion_disc(base)
    except Exception:
        section_errors.append("GEO_DISC:\n" + traceback.format_exc())

    try:
        _build_geo_particles(base)
    except Exception:
        section_errors.append("GEO_PARTICLES:\n" + traceback.format_exc())

    # ── RENDER TOP ───────────────────────────────────────────────────────────
    render = None
    try:
        render = _create_or_get_any(base, ["renderTOP"], "render1")
        _safe_set_par_any(render, ["Resolutionw", "resolutionw", "Resx", "resx"], 1280)
        _safe_set_par_any(render, ["Resolutionh", "resolutionh", "Resy", "resy"], 1280)
        # Camera parameter: try full path first, then bare name
        cam_ref = cam.path if cam is not None else "cam"
        _safe_set_par_any(render, ["Camera", "camera", "Cam", "cam_op"], cam_ref)
        _safe_set_par_any(render, ["Bgcolorr", "Backgroundcolorr", "Bgr", "bgr"], 0.0)
        _safe_set_par_any(render, ["Bgcolorg", "Backgroundcolorg", "Bgg", "bgg"], 0.0)
        _safe_set_par_any(render, ["Bgcolorb", "Backgroundcolorb", "Bgb", "bgb"], 0.0)
        _safe_set_par_any(render, ["Bgcolora", "Backgroundcolora", "Bga", "bga"], 1.0)
    except Exception:
        section_errors.append("RENDER:\n" + traceback.format_exc())

    # ── POST-FX CHAIN ─────────────────────────────────────────────────────────
    #
    #   render1 ──────────────────────→ comp_fb[0]  (current frame, foreground)
    #   feedback1 → level_fb_decay ──→ comp_fb[1]  (fading trail,  background)
    #               comp_fb ──────────→ feedback1   (closes the time-delayed loop)
    #
    #   comp_fb → displace_lens ← noise_warp
    #           → blur_glow → level_final → out1
    #
    out1 = None
    try:
        feedback1      = _create_or_get_any(base, ["feedbackTOP"],  "feedback1")
        level_fb_decay = _create_or_get_any(base, ["levelTOP"],     "level_fb_decay")
        comp_fb        = _create_or_get_any(base, ["compositeTOP"], "comp_fb")
        noise_warp     = _create_or_get_any(base, ["noiseTOP"],     "noise_warp")
        displace_lens  = _create_or_get_any(base, ["displaceTOP"],  "displace_lens")
        blur_glow      = _create_or_get_any(base, ["blurTOP"],      "blur_glow")
        level_final    = _create_or_get_any(base, ["levelTOP"],     "level_final")
        out1           = _create_or_get_any(base, ["nullTOP"],      "out1")

        _disconnect_all_inputs(comp_fb)
        _disconnect_all_inputs(level_fb_decay)
        _disconnect_all_inputs(feedback1)
        _disconnect_all_inputs(displace_lens)
        _disconnect_all_inputs(blur_glow)
        _disconnect_all_inputs(level_final)
        _disconnect_all_inputs(out1)

        if render is not None:
            comp_fb.inputConnectors[0].connect(render)
        level_fb_decay.inputConnectors[0].connect(feedback1)
        comp_fb.inputConnectors[1].connect(level_fb_decay)
        # feedbackTOP in TD requires an input connection — it internally handles
        # the 1-frame delay that breaks the cook loop.  Wire comp_fb → feedback1.
        feedback1.inputConnectors[0].connect(comp_fb)

        _safe_set_menu_any(comp_fb,
                           ["Operand", "operand", "Operation", "operation"], ["over"])
        _safe_set_par_any(comp_fb,
                          ["Operand", "operand", "Operation", "operation"], 0)

        displace_lens.inputConnectors[0].connect(comp_fb)
        displace_lens.inputConnectors[1].connect(noise_warp)
        blur_glow.inputConnectors[0].connect(displace_lens)
        level_final.inputConnectors[0].connect(blur_glow)
        out1.inputConnectors[0].connect(level_final)

        # ── AUDIO-REACTIVE EXPRESSIONS ────────────────────────────────────────
        a = "abs(op('audio_env')[0])"

        _safe_set_expr_any(level_fb_decay,
                           ["Brightness", "Brightnessx", "Brightnessy", "Brightnessz",
                            "brightness", "Gamma", "gamma", "Colr", "Colg", "Colb"],
                           "0.90 - min(0.18, {a} * 2.8)".format(a=a))

        _safe_set_par_any(noise_warp, ["Monochrome", "monochrome"], 1)
        _safe_set_par_any(noise_warp, ["Period", "period", "Freq", "freq"], 4.2)
        _safe_set_par_any(noise_warp, ["Amp", "amp", "Gain", "gain"], 0.28)
        _safe_set_par_any(noise_warp, ["Resolutionw", "resolutionw"], 1280)
        _safe_set_par_any(noise_warp, ["Resolutionh", "resolutionh"], 1280)
        _safe_set_expr_any(noise_warp,
                           ["Tx", "tx", "Translatex", "translatex", "Offsetx", "offsetx"],
                           "absTime.seconds * 0.11")
        _safe_set_expr_any(noise_warp,
                           ["Tz", "tz", "Translatez", "translatez", "Offsetz", "offsetz"],
                           "absTime.seconds * 0.07")

        lens_expr = "0.012 + min(0.28, {a} * 5.0)".format(a=a)
        _safe_set_expr_any(displace_lens,
                           ["Displaceweight", "Displaceweight1",
                            "displaceweight", "displaceweight1"],
                           lens_expr)
        for par in ["Displaceweight1", "Displaceweight2", "Displaceweight3",
                    "displaceweight1", "displaceweight2", "displaceweight3"]:
            _safe_set_expr(displace_lens, par, lens_expr)

        _safe_set_expr_any(blur_glow,
                           ["Filtersize", "Filterwidth", "Filterw", "Size",
                            "filtersize", "filterwidth", "filterw", "size"],
                           "2.0 + min(80.0, {a} * 900.0)".format(a=a))

        _safe_set_expr_any(level_final,
                           ["Brightness", "Brightnessx", "brightness"],
                           "0.82 + min(0.45, {a} * 7.0)".format(a=a))

        _set_viewer(out1, True)
    except Exception:
        section_errors.append("POST_FX:\n" + traceback.format_exc())

    # ── AUTO-LAYOUT ───────────────────────────────────────────────────────────
    # layoutChildren on the base COMP so every node is reachable with H (Home).
    _layout_children(base)

    return base, section_errors


# ─── entry point ─────────────────────────────────────────────────────────────

def build():
    project = op("/project1")
    if project is None:
        raise RuntimeError("Could not find /project1")

    try:
        if CLEAN_REBUILD:
            _destroy_if_exists(project, "blackhole_live")
            _destroy_if_exists(project, "blackhole_build_log")

        base, section_errors = _build_blackhole(project)
        _layout_children(project)

        status = "PARTIAL — see section errors below" if section_errors else "OK (all sections)"
        error_block = (
            "\n\n─── SECTION ERRORS ─────────────────────────────────────────────\n"
            + "\n\n".join(section_errors)
        ) if section_errors else ""

        _write_build_log(
            project,
            "=== Voidstar Black Hole — build {status} ===\n\n"
            "Network root : /project1/blackhole_live\n"
            "Press H inside blackhole_live to home/fit the full network.\n"
            "out1 (final output) is the rightmost TOP node.\n\n"
            "─── AUDIO ───────────────────────────────────────────────────────\n"
            "  audio_src_switch  Index=0  → live  (loopback + instrument)\n"
            "  audio_src_switch  Index=1  → file  (set File param on audio_file)\n"
            "  Run audio_diag DAT to verify input levels.\n\n"
            "─── 3D SCENE ────────────────────────────────────────────────────\n"
            "  cam               Ty=7  Tz=14  Rx=-26  (elevated angled view)\n"
            "  light_key         warm  (1.0 / 0.88 / 0.62)  intensity 0.9\n"
            "  light_fill        cool  (0.18 / 0.22 / 0.52)  intensity 0.3\n"
            "  geo_event_horizon black sphere — radius pulses with audio\n"
            "  geo_disc          torus with noise turbulence — colour driven by audio\n"
            "  geo_particles     circle ring — radius + turbulence driven by audio\n"
            "  render1           black background, camera = cam\n\n"
            "─── POST FX ─────────────────────────────────────────────────────\n"
            "  feedback trail    decay driven by audio\n"
            "  displace_lens     gravitational lensing warp\n"
            "  blur_glow         bloom corona driven by audio\n"
            "  out1              ← enable viewer here to see the simulation\n\n"
            "─── TIPS ────────────────────────────────────────────────────────\n"
            "  • CLEAN_REBUILD = {cr}\n"
            "  • Add a kinectCHOP and route hand x/y into cam.Tx / cam.Ty\n"
            "    to enable spatial interaction.\n"
            "  • Pipe out1 into a Movie File Out TOP to record."
            "{err}".format(status=status, cr=str(CLEAN_REBUILD), err=error_block)
        )

        msg = "Network: /project1/blackhole_live\nPress H to fit view.\nEnable viewer on out1 to see the simulation."
        if section_errors:
            msg += "\n\nPartial build — check blackhole_build_log for section errors."
        try:
            ui.messageBox("Voidstar Black Hole", msg)
        except Exception:
            pass

        return base

    except Exception:
        err = traceback.format_exc()
        print(err)
        _write_build_log(project, "BUILD FAILED:\n\n" + err)
        try:
            ui.messageBox(
                "Black hole build failed",
                "Full traceback written to /project1/blackhole_build_log\n"
                "(copy directly from the Text DAT).",
            )
        except Exception:
            pass
        return None


build()
