"""Diagnostic oracle ablation of the expert's hidden state; see learning/privileged_ablation.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from robot_learning.learning.privileged_ablation import main

if __name__ == '__main__':
    sys.exit(main())
