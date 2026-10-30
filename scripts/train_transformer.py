"""Train the causal-transformer policy variant; see learning/transformer_policy.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from robot_learning.learning.transformer_policy import main

if __name__ == '__main__':
    sys.exit(main())
