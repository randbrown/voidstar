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

UPDATE_AUDIO_SCALING     = False   # overwrite audio-reactivity multipliers
UPDATE_CAM_ROTATION      = False   # change cam_rotate angle (Flip TOP)
UPDATE_TITLE_TEXT        = False   # change title_hook text content
UPDATE_LOGO_PATH         = False   # change dvdlogo_png file path
ADD_COLOUR_CORRECT       = False   # insert a Colour TOP after level_drive


# ── UPDATE PARAMETERS ───────────────────────────────────────────────
# Only relevant when the matching UPDATE_* flag is True.

AUDIO_GAMMA_MULT   = 300.0      # gamma  = 1.0 + min(2.5,  env * THIS)
AUDIO_DISPLACE_MULT = 1200.0    # weight = 0.01 + min(0.8, env * THIS)
AUDIO_BLUR_MULT    = 12000.0    # blur   = 1.0 + min(120,  env * THIS)

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

def _update_audio_scaling(live):
    """Overwrite audio-reactivity expression multipliers."""
    env_expr = "abs(op('audio_env')[0])"
    gamma_expr   = "1.0 + min(2.5, ("  + env_expr + " * " + str(AUDIO_GAMMA_MULT)   + "))"
    displace_expr = "0.01 + min(0.8, (" + env_expr + " * " + str(AUDIO_DISPLACE_MULT) + "))"
    blur_expr    = "1.0 + min(120.0, (" + env_expr + " * " + str(AUDIO_BLUR_MULT)    + "))"

    results = {}

    level = live.op("level_drive")
    if level:
        results["gamma"] = _safe_set_expr_any(level, ["Gamma", "Gamma1", "gamma", "gamma1"], gamma_expr)

    displace = live.op("displace1")
    if displace:
        results["displace"] = _safe_set_expr_any(
            displace,
            ["Displaceweight", "Displaceweight1", "displaceweight", "displaceweight1"],
            displace_expr,
        )
        _safe_set_expr(displace, "Displaceweight2", displace_expr)
        _safe_set_expr(displace, "Displaceweight3", displace_expr)
        _safe_set_expr(displace, "displaceweight2", displace_expr)
        _safe_set_expr(displace, "displaceweight3", displace_expr)

    blur = live.op("blur1")
    if blur:
        results["blur"] = _safe_set_expr_any(
            blur,
            ["Filtersize", "Filterwidth", "filtersize", "filterwidth"],
            blur_expr,
        )

    return results


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
        if UPDATE_AUDIO_SCALING:
            any_enabled = True
            r = _update_audio_scaling(live)
            log_lines.append("Audio scaling: " + str(r))

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
