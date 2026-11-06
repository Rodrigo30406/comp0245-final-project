"""Load optional local settings without adding dependencies to RoboEnv."""

import os
from pathlib import Path
import shlex


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_roboenv_root():
    """Resolve the shared simulator assets, independently of the terminal cwd."""
    configured = os.environ.get("ROBOENV_ROOT")
    root = Path(configured).expanduser() if configured else PROJECT_ROOT.parent / "RoboEnv"
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    if not (root / "configs" / "robotnik_arm.json").is_file() or not (root / "models").is_dir():
        raise FileNotFoundError(
            f"RoboEnv assets not found at {root}. Set ROBOENV_ROOT to the RoboEnv "
            "repository root containing configs/robotnik_arm.json and models/."
        )
    return str(root)


def load_project_env(env_file=None):
    """Read single-line KEY=value settings; existing environment values win.

    Quoted values and comments are supported. Shell execution, interpolation,
    and multiline values are deliberately not part of this configuration format.
    """
    env_file = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line_number, line in enumerate(env_file.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not key.isidentifier():
            raise ValueError(f"Invalid assignment in {env_file}:{line_number}; use KEY=value.")
        try:
            values = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as error:
            raise ValueError(f"Invalid quoting in {env_file}:{line_number}.") from error
        if len(values) > 1:
            raise ValueError(f"Quote values containing spaces in {env_file}:{line_number}.")
        os.environ.setdefault(key, values[0] if values else "")
