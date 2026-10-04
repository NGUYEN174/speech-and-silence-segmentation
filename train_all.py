"""
Training pipeline for Speech/Silence Segmentation.

Usage:
    python train_all.py

This script:
    1. Finds all training WAV files and matching LAB files
    2. Validates all training data pairs
    3. Trains each algorithm using ONLY training data
    4. Saves trained parameters to config/trained_parameters.json

IMPORTANT:
    - ONLY files in data/TinHieuHuanLuyen/ are used for training.
    - Test data (data/TinHieuKiemThu/) is NEVER accessed during training.
    - Each algorithm's training is independently callable.
"""

import os
import sys
import glob

from config.parameters import (
    TRAINING_DIR, FRAME_LENGTH_MS, FRAME_SHIFT_MS, MIN_SILENCE_MS,
    save_trained_parameters
)
from src.common.interfaces import get_all_algorithms


def find_training_pairs(training_dir):
    """Find all WAV/LAB file pairs in the training directory.
    
    Args:
        training_dir (str): Path to training data directory.
    
    Returns:
        tuple: (wav_files, lab_files) - sorted lists of matching paths.
    
    Raises:
        FileNotFoundError: If directory doesn't exist or no files found.
        ValueError: If WAV/LAB pairs don't match.
    """
    if not os.path.exists(training_dir):
        print(f"\nERROR: Training directory not found:")
        print(f"  {training_dir}")
        print(f"\nPlease create the directory and place the lecturer's")
        print(f"training files (*.wav and *.lab) there.")
        sys.exit(1)
    
    # Find WAV files
    wav_files = sorted(glob.glob(os.path.join(training_dir, '*.wav')))
    
    if not wav_files:
        print(f"\nERROR: No training WAV files found.")
        print(f"\nPlease place the lecturer's training files in:")
        print(f"  {training_dir}")
        sys.exit(1)
    
    # Find matching LAB files
    lab_files = []
    for wav_path in wav_files:
        lab_path = os.path.splitext(wav_path)[0] + '.lab'
        if not os.path.exists(lab_path):
            print(f"\nERROR: Missing LAB file for {os.path.basename(wav_path)}")
            print(f"  Expected: {lab_path}")
            sys.exit(1)
        lab_files.append(lab_path)
    
    return wav_files, lab_files


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  SPEECH/SILENCE SEGMENTATION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Print fixed parameters
    print(f"\nFixed parameters (NOT tuned):")
    print(f"  Frame length : {FRAME_LENGTH_MS} ms")
    print(f"  Frame shift  : {FRAME_SHIFT_MS} ms")
    print(f"  Min silence  : {MIN_SILENCE_MS} ms")
    
    # Find training data
    print(f"\nSearching for training data in:")
    print(f"  {TRAINING_DIR}")
    wav_files, lab_files = find_training_pairs(TRAINING_DIR)
    print(f"\n  Found {len(wav_files)} training file pairs:")
    for wav in wav_files:
        print(f"    {os.path.basename(wav)}")
    
    # Train each algorithm
    all_params = {}
    algorithms = get_all_algorithms()
    
    for algo in algorithms:
        print(f"\n{'=' * 50}")
        print(f"  Training: {algo.name}")
        print(f"{'=' * 50}")
        
        params = algo.train(wav_files, lab_files)
        all_params[algo.key] = params
    
    # Save all trained parameters
    print(f"\n{'=' * 50}")
    print(f"  SAVING PARAMETERS")
    print(f"{'=' * 50}")
    save_trained_parameters(all_params)
    
    # Summary
    print(f"\n{'=' * 50}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 50}")
    print(f"\nTrained parameters for {len(all_params)} algorithms.")
    for key, params in all_params.items():
        if params:
            print(f"\n  [{key}]")
            for p_name, p_value in params.items():
                print(f"    {p_name}: {p_value}")
        else:
            print(f"\n  [{key}] - placeholder (no parameters)")
    
    print(f"\nNext step: run 'python run_all.py' for testing.")


if __name__ == '__main__':
    main()
