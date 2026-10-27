"""Launch paired autonomous evaluation using the active RoboEnv environment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robot_learning.evaluation.cli import main

if __name__=='__main__':
    sys.exit(main())
