# main_phase5.py
"""
Phase 5: Evaluation & Before/After Testing
Run this after training is complete
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import ModelEvaluator


def main():
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "PHASE 5: EVALUATION".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝\n")

    evaluator = ModelEvaluator()
    evaluator.run_evaluation()

    print("\n Evaluation complete!")
    print("Results: results/")


if __name__ == "__main__":
    main()