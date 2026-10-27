"""Collect, train and evaluate the nested episode-count learning curve."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robot_learning.learning.scaling import main
if __name__=='__main__':main()
