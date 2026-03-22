"""
TouchDesigner builder for a Voidstar starter network.

How to use inside TouchDesigner:
1) Create a Text DAT in /project1.
2) Paste this file content.
3) Click Run Script.
4) Save project as a .toe.

This script builds:
- /project1/voidstar_live (real-time performance chain)
- /project1/voidstar_post (post-process controls and command preview)

It is intentionally conservative and uses native TOP/CHOP operators so it can run in real-time.
"""

import traceback

# When True, rerunning the builder destroys and recreates generated comps for a clean rebuild.
# Set to False if you want to preserve manual edits inside voidstar_live/voidstar_post.
CLEAN_REBUILD = True


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
    # Sets a menu parameter by matching token text against menu names/labels.
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
                for i, l in enumerate(labels):
                    if t in l:
                        p.menuIndex = i
                        return par_name
        except Exception:
            pass
    return ""


def _create_or_get(parent, cls, name):
    existing = parent.op(name)
    if existing is not None:
        return existing
    return parent.create(cls, name)


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

    raise RuntimeError("Could not create operator '" + name + "' from types: " + ", ".join(type_names))


def _layout_children(parent):
    try:
        parent.layoutChildren(horizontal=True, vertical=True)
    except Exception:
        pass


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


def _disconnect_input(op_obj, input_index):
    try:
        conn = op_obj.inputConnectors[input_index]
    except Exception:
        return
    try:
        # Explicitly clear any previous incoming wires when rebuilding in-place.
        for c in list(conn.connections):
            c.disconnect()
    except Exception:
        pass


def _write_build_log(project, message):
    log_dat = _create_or_get_any(project, ["textDAT"], "voidstar_build_log")
    log_dat.text = message
    log_dat.nodeX = -40
    log_dat.nodeY = -220
    return log_dat


def _destroy_if_exists(parent, name):
    op_obj = parent.op(name)
    if op_obj is None:
        return False
    try:
        op_obj.destroy()
        return True
    except Exception:
        return False


def _audio_diag_dat_text():
    return (
        "\"\"\"Audio diagnostics for /project1/voidstar_live.\"\"\"\n"
        "\n"
        "def _channel_absmax(chop_op):\n"
        "    if chop_op is None:\n"
        "        return 0.0\n"
        "    vals = []\n"
        "    try:\n"
        "        chans = list(chop_op.chans())\n"
        "    except Exception:\n"
        "        chans = []\n"
        "    for ch in chans:\n"
        "        try:\n"
        "            vals.append(abs(ch.eval()))\n"
        "        except Exception:\n"
        "            pass\n"
        "    return max(vals) if vals else 0.0\n"
        "\n"
        "def report_audio_status(parent_path='/project1/voidstar_live', threshold=0.001):\n"
        "    parent = op(parent_path)\n"
        "    if parent is None:\n"
        "        print('[RED] Missing ' + parent_path)\n"
        "        return {'ok': False, 'reason': 'missing parent'}\n"
        "\n"
        "    inst = parent.op('audio_instrument')\n"
        "    loop = parent.op('audio_loopback')\n"
        "    env = parent.op('audio_env')\n"
        "\n"
        "    inst_val = _channel_absmax(inst)\n"
        "    loop_val = _channel_absmax(loop)\n"
        "    env_val = _channel_absmax(env)\n"
        "\n"
        "    inst_ok = inst_val > threshold\n"
        "    loop_ok = loop_val > threshold\n"
        "    env_ok = env_val > threshold\n"
        "\n"
        "    print('Voidstar audio diagnostics')\n"
        "    print(('[GREEN] ' if inst_ok else '[RED] ') + 'audio_instrument max=' + ('%.6f' % inst_val))\n"
        "    print(('[GREEN] ' if loop_ok else '[RED] ') + 'audio_loopback   max=' + ('%.6f' % loop_val))\n"
        "    print(('[GREEN] ' if env_ok else '[RED] ') + 'audio_env        max=' + ('%.6f' % env_val))\n"
        "\n"
        "    overall_ok = inst_ok or loop_ok\n"
        "    print(('[GREEN] ' if overall_ok else '[RED] ') + 'overall input status')\n"
        "\n"
        "    return {\n"
        "        'instrument_ok': inst_ok,\n"
        "        'loopback_ok': loop_ok,\n"
        "        'env_ok': env_ok,\n"
        "        'instrument_value': inst_val,\n"
        "        'loopback_value': loop_val,\n"
        "        'env_value': env_val,\n"
        "        'ok': overall_ok,\n"
        "    }\n"
        "\n"
        "# Running this DAT prints one status snapshot in Textport.\n"
        "report_audio_status()\n"
    )


def _build_live(parent_comp):
    base = _create_or_get_any(parent_comp, ["baseCOMP"], "voidstar_live")
    base.nodeX = -350
    base.nodeY = 120

    # Source inputs
    cam_in = _create_or_get_any(base, ["videodeviceinTOP"], "cam_in")
    cam_rotate = _create_or_get_any(base, ["flipTOP"], "cam_rotate")
    movie_in = _create_or_get_any(base, ["moviefileinTOP"], "movie_in")
    src_switch = _create_or_get_any(base, ["switchTOP"], "src_switch")
    src_null = _create_or_get_any(base, ["nullTOP"], "src")

    cam_rotate.inputConnectors[0].connect(cam_in)
    src_switch.inputConnectors[0].connect(cam_rotate)
    src_switch.inputConnectors[1].connect(movie_in)
    src_null.inputConnectors[0].connect(src_switch)

    # Audio CHOP chain (instrument + loopback)
    audio_inst = _create_or_get_any(base, ["audiodevinCHOP", "audiodeviceinCHOP"], "audio_instrument")
    audio_loop = _create_or_get_any(base, ["audiodevinCHOP", "audiodeviceinCHOP"], "audio_loopback")
    audio_merge = _create_or_get_any(base, ["mergeCHOP"], "audio_merge")
    audio_analyze = _create_or_get_any(base, ["analyzeCHOP"], "audio_analyze")
    audio_math = _create_or_get_any(base, ["mathCHOP"], "audio_math")
    audio_filter = _create_or_get_any(base, ["filterCHOP"], "audio_filter")
    audio_env = _create_or_get_any(base, ["nullCHOP"], "audio_env")

    # Keep loopback first (stereo software/audio-bus feed), instrument second.
    audio_merge.inputConnectors[0].connect(audio_loop)
    audio_merge.inputConnectors[1].connect(audio_inst)
    audio_analyze.inputConnectors[0].connect(audio_merge)
    audio_math.inputConnectors[0].connect(audio_analyze)
    audio_filter.inputConnectors[0].connect(audio_math)
    audio_env.inputConnectors[0].connect(audio_filter)

    _safe_set_par(audio_analyze, "Function", "rms")
    _safe_set_par(audio_filter, "Filterwidth", 0.08)
    _safe_set_par(audio_env, "Cooktype", "always")

    # Flip TOP handles rotation in 90-degree increments and auto-swaps output resolution.
    # Flip rotate: 0=0deg, 1=90deg, 2=180deg, 3=270deg (menu parameter).
    _safe_set_menu_any(cam_rotate, ["Rotate", "rotate"], ["270", "3"])
    # Fallback: try setting as integer index directly (3 = 270 degrees).
    _safe_set_par_any(cam_rotate, ["Rotate", "rotate"], 3)

    # Try to force loopback source to stereo where available.
    _safe_set_par_any(audio_loop, ["Numchans", "numchans", "Numchannels", "numchannels", "Chans", "chans"], 2)

    audio_diag = _create_or_get_any(base, ["textDAT"], "audio_diag")
    audio_diag.text = _audio_diag_dat_text()

    # Visual chain (no feedback; stable baseline)
    level_drive = _create_or_get_any(base, ["levelTOP"], "level_drive")
    edge = _create_or_get_any(base, ["edgeTOP"], "edge1")
    blur = _create_or_get_any(base, ["blurTOP"], "blur1")
    noise = _create_or_get_any(base, ["noiseTOP"], "noise1")
    displace = _create_or_get_any(base, ["displaceTOP"], "displace1")

    logo = _create_or_get_any(base, ["moviefileinTOP"], "dvdlogo_png")
    logo_over = _create_or_get_any(base, ["compositeTOP"], "logo_over")

    title = _create_or_get_any(base, ["textTOP"], "title_hook")
    final_over = _create_or_get_any(base, ["compositeTOP"], "title_over")
    out_null = _create_or_get_any(base, ["nullTOP"], "out1")

    level_drive.inputConnectors[0].connect(src_null)
    edge.inputConnectors[0].connect(level_drive)
    blur.inputConnectors[0].connect(edge)

    displace.inputConnectors[0].connect(level_drive)
    displace.inputConnectors[1].connect(noise)

    # Composite TOP: Input 0 = background, Input 1 = overlay.
    logo_over.inputConnectors[0].connect(displace)
    logo_over.inputConnectors[1].connect(logo)

    final_over.inputConnectors[0].connect(logo_over)
    final_over.inputConnectors[1].connect(title)
    out_null.inputConnectors[0].connect(final_over)

    # Composite TOP: set operation to "Over" for alpha-based compositing.
    for comp_top in [logo_over, final_over]:
        _safe_set_menu_any(comp_top, ["Operand", "operand", "Operation", "operation"], ["over"])
        _safe_set_par_any(comp_top, ["Operand", "operand", "Operation", "operation"], 0)

    # Deterministic layout so Home (h) consistently frames the full network.
    _set_node_pos(cam_in, -1100, 200)
    _set_node_pos(cam_rotate, -940, 200)
    _set_node_pos(movie_in, -1100, 20)
    _set_node_pos(src_switch, -780, 110)
    _set_node_pos(src_null, -620, 110)

    _set_node_pos(audio_inst, -1120, -260)
    _set_node_pos(audio_loop, -1120, -420)
    _set_node_pos(audio_merge, -900, -340)
    _set_node_pos(audio_analyze, -700, -340)
    _set_node_pos(audio_math, -520, -340)
    _set_node_pos(audio_filter, -340, -340)
    _set_node_pos(audio_env, -160, -340)
    _set_node_pos(audio_diag, -160, -520)

    _set_node_pos(level_drive, -520, 110)
    _set_node_pos(edge, -320, 110)
    _set_node_pos(blur, 60, 110)
    _set_node_pos(noise, 60, -40)
    _set_node_pos(displace, 260, 110)
    _set_node_pos(logo, 260, -70)
    _set_node_pos(logo_over, 460, 110)
    _set_node_pos(title, 460, -70)
    _set_node_pos(final_over, 660, 110)
    _set_node_pos(out_null, 860, 110)

    _set_viewer(out_null, True)

    # Starter visual defaults
    _safe_set_par_any(movie_in, ["Play", "play"], 1)
    _safe_set_par_any(noise, ["Monochrome", "monochrome"], 1)
    _safe_set_par_any(displace, ["Displaceweight", "displaceweight", "Displaceweight1", "displaceweight1"], 0.02)
    _safe_set_par_any(logo, ["File", "file"], "../dvd_logo/voidstar_logo_0.png")
    _safe_set_par_any(title, ["Text", "text"], "VOIDSTAR LIVE")
    _safe_set_par_any(title, ["Resolutionw", "resolutionw"], 1920)
    _safe_set_par_any(title, ["Resolutionh", "resolutionh"], 1080)

    # Explicit CHOP->TOP expressions are more reliable across TD builds than channel export calls.
    # This uses first channel from audio_env and applies strong starter scaling.
    audio_expr = "abs(op('audio_env')[0])"
    gamma_expr = "1.0 + min(2.5, (" + audio_expr + " * 300.0))"
    displace_expr = "0.01 + min(0.8, (" + audio_expr + " * 1200.0))"
    blur_expr = "1.0 + min(120.0, (" + audio_expr + " * 12000.0))"

    gamma_par = _safe_set_expr_any(level_drive, ["Gamma", "Gamma1", "gamma", "gamma1"], gamma_expr)
    displace_par = _safe_set_expr_any(
        displace,
        ["Displaceweight", "Displaceweight1", "Weight", "Displace", "displaceweight", "displaceweight1", "weight", "displace"],
        displace_expr,
    )
    blur_par = _safe_set_expr_any(blur, ["Filtersize", "Filterwidth", "Filterw", "Size", "filtersize", "filterwidth", "filterw", "size"], blur_expr)

    # For Displace TOP, apply to RGB displacement weights when present.
    _safe_set_expr(displace, "Displaceweight1", displace_expr)
    _safe_set_expr(displace, "Displaceweight2", displace_expr)
    _safe_set_expr(displace, "Displaceweight3", displace_expr)
    _safe_set_expr(displace, "displaceweight1", displace_expr)
    _safe_set_expr(displace, "displaceweight2", displace_expr)
    _safe_set_expr(displace, "displaceweight3", displace_expr)

    expr_report = {
        "level_gamma": gamma_par,
        "displace_weight": displace_par,
        "blur_size": blur_par,
    }

    return base, expr_report


def _build_post(parent_comp):
    base = _create_or_get_any(parent_comp, ["baseCOMP"], "voidstar_post")
    base.nodeX = 250
    base.nodeY = 120

    note = _create_or_get_any(base, ["textDAT"], "README")
    note.text = (
        "voidstar_post\n"
        "1) Export a clip from /project1/voidstar_live/out1 with Movie File Out TOP.\n"
        "2) Run the command in command_preview DAT from terminal or a DAT script.\n"
        "3) Effects supported: dvdlogo, glitchfield, reels_overlay, title_hook.\n"
    )

    command_preview = _create_or_get_any(base, ["textDAT"], "command_preview")
    command_preview.text = (
        "python3 touchdesigner/voidstar_td_starter/voidstar_post_bridge.py "
        "--effect dvdlogo "
        "--input /absolute/path/to/export.mp4 "
        "--output /absolute/path/to/export_dvdlogo.mp4 "
        "--logo dvd_logo/voidstar_logo_0.png"
    )

    _set_node_pos(note, -100, 60)
    _set_node_pos(command_preview, 240, 60)
    return base


def build():
    project = op("/project1")
    if project is None:
        raise RuntimeError("Could not find /project1")

    try:
        if CLEAN_REBUILD:
            _destroy_if_exists(project, "voidstar_live")
            _destroy_if_exists(project, "voidstar_post")
            _destroy_if_exists(project, "voidstar_build_log")

        live, expr_report = _build_live(project)
        post = _build_post(project)
        _layout_children(project)
        _write_build_log(
            project,
            "Build succeeded.\n"
            "Created /project1/voidstar_live and /project1/voidstar_post.\n"
            "Run /project1/voidstar_live/audio_diag for input checks.\n"
            "Audio->visual mapping: expression mode (strong scaling).\n"
            "Expression bind level gamma: " + str(expr_report.get("level_gamma") or "FAILED") + "\n"
            "Expression bind displace weight: " + str(expr_report.get("displace_weight") or "FAILED") + "\n"
            "Expression bind blur size: " + str(expr_report.get("blur_size") or "FAILED") + "\n"
            "Audio merge order: audio_loopback first, audio_instrument second.\n"
            "Camera rotate control: /project1/voidstar_live/cam_rotate (Flip TOP, 270 deg).\n"
            "Feedback module: disabled (stable baseline mode).\n"
            "Clean rebuild mode: " + str(CLEAN_REBUILD) + "\n"
            "Children in voidstar_live: " + str(len(op('/project1/voidstar_live').children)),
        )

        ui.messageBox(
            "Voidstar starter created",
            "Created /project1/voidstar_live and /project1/voidstar_post.\n"
            "Now configure audio devices and save your project as a .toe.",
        )
        return live, post
    except Exception:
        err = traceback.format_exc()
        print(err)
        _write_build_log(project, err)
        try:
            ui.messageBox(
                "Voidstar starter build failed",
                "Traceback written to /project1/voidstar_build_log (copyable Text DAT).",
            )
        except Exception:
            pass
        return None, None


build()
