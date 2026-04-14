
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FEW_SHOT_DIR = DATA_DIR / "few_shot"
PROCESSED_DIR = DATA_DIR / "processed"

CLASS_MAPPING = {
    1: "Speed limit 30 km/h",
    2: "Speed limit 50 km/h",
    5: "Speed limit 80 km/h",
    14: "Stop",
    17: "No entry"
}

IMAGE_SIZE = 32
TRAIN_IMAGES_PER_CLASS = 6
REFERENCE_IMAGES_PER_CLASS = 2