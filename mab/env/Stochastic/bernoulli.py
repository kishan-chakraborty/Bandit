from pathlib import Path
import sys

current_dir = Path.cwd()
sys.path.append(str(current_dir.parent))
from mab.env.Stochastic.env import env