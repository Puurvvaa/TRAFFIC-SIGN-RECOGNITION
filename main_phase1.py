# main_phase1.py
"""
Main script for Phase 1: Data Preparation
Run this first after manual folder setup
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preparation import DataPreparation


def main():
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "PHASE 1: DATA PREPARATION".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝\n")

    prep = DataPreparation()
    success = prep.run_all()

    if success:
        print("\n Dataset ready!")
        print("Location: data/processed/")
    else:
        print("\n Setup incomplete")
        sys.exit(1)


if __name__ == "__main__":
    main()