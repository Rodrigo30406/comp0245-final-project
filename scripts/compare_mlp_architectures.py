"""Launch frozen-source depth/shape comparisons with existing reference models."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robot_learning.learning.architecture_comparison import main

if __name__=='__main__':main()
