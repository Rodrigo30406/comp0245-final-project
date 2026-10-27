"""Export trajectory and fidelity diagnostics from an episode-count campaign."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from robot_learning.evaluation.size_report import main

if __name__ == '__main__':
    main()
