import os
from pathlib import Path


# Directories
BASE_DIR = Path(os.path.abspath(__file__)).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
MODEL_DIR = Path(BASE_DIR, "model")