import sys
import os
import numpy as np

SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')

sys.path.insert(0, SRC_DIR)
from disjoint_context import DisjointContextRecommender

def run_validations():
    print("Initializing Recommender System for Validation...")
    recommender = DisjointContextRecommender(
        output_dir=MODELS_DIR,
        input_dir=DATA_DIR,
        config_dir=CONFIG_DIR,
    )

    print("\n[Validation A] Testing Known Substitutions (Expect Substitute Status)")
    substitutes_to_test = [
        ("beef", "lamb"),
        ("lime", "lemon"),
        ("butter", "margarine"),
        ("sugar", "honey")
    ]

    for ing1, ing2 in substitutes_to_test:
        print(f"\nTesting {ing1} -> {ing2}...")
        try:
            recommender.check_pair(ing1, ing2)
        except Exception as e:
            print(f"Error checking pair: {e}")

    print("\n[Validation B] Testing Known Complements (Expect COMPLEMENT Status)")
    complements_to_test = [
        ("butter", "bread"),
        ("beef", "wine"),
        ("tomato", "basil")
    ]

    for ing1, ing2 in complements_to_test:
        print(f"\nTesting {ing1} -> {ing2}...")
        recommender.check_pair(ing1, ing2)

    print("\n[Validation C] Testing Unrelated Pairs (Expect Low Sim)")
    unrelated = [
        ("beef", "vanilla"),
        ("fish", "chocolate")
    ]
    for ing1, ing2 in unrelated:
        print(f"Testing {ing1} -> {ing2}...")
        recommender.check_pair(ing1, ing2)

if __name__ == "__main__":
    run_validations()
