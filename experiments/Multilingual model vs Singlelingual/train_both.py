"""
Script to train both models sequentially:
1. Train multilingual model (500k reviews across languages)
2. Train single language model (500k reviews from one language)
"""

import subprocess
import sys
import os

print("=" * 60)
print("Training Both Models")
print("=" * 60)
print("\nThis will train two models:")
print("1. Multilingual model (500k reviews across languages)")
print("2. Single language model (500k reviews from one language)")
print("\nThis may take a while...")

# Train multilingual model
print("\n" + "=" * 60)
print("TRAINING MODEL 1: MULTILINGUAL")
print("=" * 60)
result1 = subprocess.run([sys.executable, "main.py", "multilingual"], 
                        capture_output=False)

if result1.returncode != 0:
    print(f"\nError training multilingual model (exit code: {result1.returncode})")
    sys.exit(1)

# Train single language model
print("\n" + "=" * 60)
print("TRAINING MODEL 2: SINGLE LANGUAGE")
print("=" * 60)
result2 = subprocess.run([sys.executable, "main.py", "single_language"], 
                        capture_output=False)

if result2.returncode != 0:
    print(f"\nError training single language model (exit code: {result2.returncode})")
    sys.exit(1)

print("\n" + "=" * 60)
print("Both models trained successfully!")
print("=" * 60)
print("\nResults:")
print(f"  - Multilingual model: ./results_multilingual/")
print(f"  - Single language model: ./results_single_language/")
print("\nPlots:")
print(f"  - Multilingual: ./results_multilingual/training_validation_loss_multilingual.png")
print(f"  - Single language: ./results_single_language/training_validation_loss_single_language.png")
