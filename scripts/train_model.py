"""Train a supervised baseline using the active RoboEnv Python environment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from robot_learning.learning.train import main

if __name__ == '__main__':
    sys.exit(main())
