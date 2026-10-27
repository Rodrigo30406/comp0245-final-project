"""Run the patience-only MLP extension through 2048 episodes per design."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robot_learning.learning.patience_series import main

if __name__=='__main__':
    main()
