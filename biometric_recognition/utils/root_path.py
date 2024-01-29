import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = Path(script_dir).parent
root = str(script_dir)