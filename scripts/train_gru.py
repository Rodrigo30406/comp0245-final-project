"""Train the gru policy; see src/robot_learning/learning/gru.py and docs/03_models.md. Run with the RoboEnv ml environment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_learning.learning.gru import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
