# main_phase3_4.py
"""
Phases 3 & 4: Model Building & Training
Run this after Phase 2 is complete
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.training import SiameseTrainer


def main():
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "PHASES 3 & 4: MODEL & TRAINING".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝\n")

    trainer = SiameseTrainer()
    model = trainer.train(epochs=50, batch_size=32)

    print("\n Model trained and saved!")
    print(" Location: models/siamese_model.h5")


if __name__ == "__main__":
    main()