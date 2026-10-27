"""Report command tolerances for frozen baselines using the RoboEnv environment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robot_learning.learning.report import main

if __name__=='__main__':sys.exit(main())
