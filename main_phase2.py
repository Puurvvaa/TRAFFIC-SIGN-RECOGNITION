# main_phase2.py
"""
Phase 2: Pair Generation
Run this after Phase 1 is complete
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pair_generation import PairGenerator


def main():
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "PHASE 2: PAIR GENERATION".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝\n")

    generator = PairGenerator()
    success = generator.run_all()

    if success:
        print("\n Pairs generated!")
        print(" Location: data/pairs/")


if __name__ == "__main__":
    main()