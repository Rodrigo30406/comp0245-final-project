"""Train the declared egocentric-plus-clock MLP variant; see learning/egocentric.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from robot_learning.learning.egocentric import main

if __name__ == '__main__':
    sys.exit(main())
