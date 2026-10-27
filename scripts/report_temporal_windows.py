"""Report complete paired temporal-policy trajectory distributions."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robot_learning.evaluation.temporal_report import main

if __name__=='__main__':main()
