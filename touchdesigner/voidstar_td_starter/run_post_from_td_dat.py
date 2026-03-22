"""
TouchDesigner DAT helper for launching offline Voidstar post-processing.

Usage inside TouchDesigner textport or DAT:

mod_path = project.folder + '/touchdesigner/voidstar_td_starter/run_post_from_td_dat.py'
exec(open(mod_path, 'r', encoding='utf-8').read())
run_post(effect='dvdlogo', input_path='/abs/in.mp4', output_path='/abs/out.mp4')
"""

import subprocess
from pathlib import Path


def _resolve_python(root: Path) -> str:
    env_python = root / ".venv_voidstar_td" / "bin" / "python"
    if env_python.exists():
        return str(env_python)
    return "python3"


def run_post(effect, input_path, output_path, extra=None, root_path=None):
    if root_path:
        root = Path(root_path)
    elif "project" in globals() and hasattr(project, "folder"):
        root = Path(project.folder)
    else:
        root = Path.cwd()
    bridge = root / "touchdesigner" / "voidstar_td_starter" / "voidstar_post_bridge.py"
    py = _resolve_python(root)

    cmd = [
        py,
        str(bridge),
        "--effect",
        str(effect),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]

    if extra:
        cmd.extend(["--extra", str(extra)])

    print("[voidstar-td] launching:", " ".join(cmd))
    return subprocess.Popen(cmd)
