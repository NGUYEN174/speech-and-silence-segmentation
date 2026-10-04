"""
Fixed and configurable parameters for the Speech/Silence Segmentation project.

FIXED PARAMETERS (set by the lecturer - MUST NOT be optimized):
    FRAME_LENGTH_MS = 25 ms
    FRAME_SHIFT_MS  = 10 ms
    MIN_SILENCE_MS  = 200 ms

These values are assignment constraints and must remain constant
during both training and testing.
"""

import os
import json

# ============================================================
# FIXED PARAMETERS (DO NOT MODIFY)
# These are set by the lecturer as assignment constraints.
# ============================================================

FRAME_LENGTH_MS = 25    # Frame length in milliseconds
FRAME_SHIFT_MS = 10     # Frame shift (hop) in milliseconds
MIN_SILENCE_MS = 200    # Minimum silence duration in milliseconds

# ============================================================
# PATHS
# ============================================================

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
TRAINING_DIR = os.path.join(PROJECT_ROOT, "data", "TinHieuHuanLuyen")
TESTING_DIR = os.path.join(PROJECT_ROOT, "data", "TinHieuKiemThu")

# Results directories
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_BINARY_SEARCH = os.path.join(RESULTS_DIR, "binary_search")
RESULTS_HISTOGRAM = os.path.join(RESULTS_DIR, "histogram")
RESULTS_SIMPLE_STATISTICS = os.path.join(RESULTS_DIR, "simple_statistics")
RESULTS_SNR = os.path.join(RESULTS_DIR, "snr")

# Trained parameters file
TRAINED_PARAMS_PATH = os.path.join(
    PROJECT_ROOT, "config", "trained_parameters.json"
)

# ============================================================
# NUMERICAL CONSTANTS
# ============================================================

# Small constant to avoid log(0)
EPSILON = 1e-12

# ============================================================
# SNR EXPERIMENT SETTINGS
# ============================================================

# Default SNR levels for robustness experiment (in dB)
SNR_LEVELS_DB = [20, 10, 5, 0]

# Random seed for reproducibility of noise generation
SNR_RANDOM_SEED = 42

# ============================================================
# ALGORITHM REGISTRY
# ============================================================

# List of all algorithm module names
ALGORITHM_NAMES = ["binary_search", "histogram", "simple_statistics"]


def load_trained_parameters():
    """Load trained parameters from JSON file.
    
    Returns:
        dict: Trained parameters for all algorithms.
    
    Raises:
        FileNotFoundError: If trained_parameters.json does not exist.
    """
    if not os.path.exists(TRAINED_PARAMS_PATH):
        raise FileNotFoundError(
            f"ERROR: trained_parameters.json not found at:\n"
            f"  {TRAINED_PARAMS_PATH}\n\n"
            f"Please run:\n"
            f"  python train_all.py\n\n"
            f"to train algorithms and generate parameters."
        )
    
    with open(TRAINED_PARAMS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_trained_parameters(params):
    """Save trained parameters to JSON file.
    
    Args:
        params (dict): Parameters dictionary to save.
    """
    os.makedirs(os.path.dirname(TRAINED_PARAMS_PATH), exist_ok=True)
    with open(TRAINED_PARAMS_PATH, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
    print(f"\nTrained parameters saved to: {TRAINED_PARAMS_PATH}")


def ensure_results_dirs():
    """Create all result directories if they do not exist."""
    for d in [RESULTS_DIR, RESULTS_BINARY_SEARCH, RESULTS_HISTOGRAM,
              RESULTS_SIMPLE_STATISTICS, RESULTS_SNR]:
        os.makedirs(d, exist_ok=True)
