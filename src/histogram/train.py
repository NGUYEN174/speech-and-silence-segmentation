"""
Histogram algorithm training module.

Determines optimal parameters for the Histogram-based speech/silence
segmentation algorithm using ONLY training data.

Trainable parameters:
    - histogram_bins: Number of histogram bins
    - smoothing_sigma: Gaussian smoothing sigma
    - W: Weighting parameter for threshold formula T = (W*M1 + M2)/(W+1)
    - peak_prominence: Minimum peak prominence (as fraction of max)
    - peak_distance: Minimum distance between peaks (in bins)

Fixed parameters (NOT tuned here):
    - Frame length: 25 ms
    - Frame shift: 10 ms
    - Minimum silence: 200 ms

Training metric:
    Boundary MAE on training data (lower is better).

IMPORTANT:
    - ONLY training WAV and LAB files are used.
    - Test LAB files are NEVER accessed during training.
    - The parameter grid is kept reasonable to avoid overfitting.
"""

import numpy as np
import itertools

from src.common.audio import load_audio
from src.common.framing import frame_signal
from src.common.features import calculate_ste, calculate_log_ste, calculate_spectral_centroid
from src.common.lab_reader import read_lab, get_speech_boundaries
from src.common.segments import mask_to_segments, segments_to_boundaries, boundaries_to_ms
from src.common.postprocess import merge_short_silences
from src.common.evaluation import evaluate_boundaries
from src.histogram.algorithm import compute_histogram_threshold, _combine_feature_decisions


# ============================================================
# Candidate parameter grid
# Keep this grid REASONABLE. Do not over-engineer.
# ============================================================

CANDIDATE_BINS = [30, 50, 70]
CANDIDATE_SIGMA = [2.0, 3.0, 5.0]
CANDIDATE_W = [3.0, 5.0, 7.0, 10.0]
CANDIDATE_PROMINENCE = [0.005, 0.01, 0.02]
CANDIDATE_DISTANCE = [3, 5, 8]


def train_histogram(training_files, training_labs):
    """Train Histogram parameters using a grid search over training data.
    
    For each candidate parameter combination:
        1. Process all training files
        2. Compute histogram thresholds for energy and spectral centroid
        3. Generate speech mask and boundaries
        4. Compare with ground-truth boundaries
        5. Compute average MAE across all training files
    
    Select the configuration with the lowest average training MAE.
    
    Args:
        training_files (list of str): Paths to training WAV files.
        training_labs (list of str): Paths to training LAB files.
    
    Returns:
        dict: Best parameters:
            {
                'histogram_bins': int,
                'smoothing_sigma': float,
                'W': float,
                'peak_prominence': float,
                'peak_distance': int,
                'training_mae_ms': float,
                'training_rmse_ms': float,
                'feature_representation': str,
            }
    """
    print("\n" + "=" * 50)
    print("HISTOGRAM TRAINING")
    print("=" * 50)
    print(f"\nFeature representation: log STE + Spectral Centroid")
    print(f"Training files: {len(training_files)}")
    print(f"\nCandidate bins: {CANDIDATE_BINS}")
    print(f"Candidate sigma: {CANDIDATE_SIGMA}")
    print(f"Candidate W: {CANDIDATE_W}")
    print(f"Candidate peak prominence: {CANDIDATE_PROMINENCE}")
    print(f"Candidate peak distance: {CANDIDATE_DISTANCE}")
    
    # Pre-load and pre-compute features for all training files
    # to avoid redundant computation
    print("\nPre-computing features for training files...")
    training_data = []
    
    for wav_path, lab_path in zip(training_files, training_labs):
        signal, fs = load_audio(wav_path)
        frame_result = frame_signal(signal, fs)
        frames = frame_result['frames']
        feature_times = frame_result['center_times']
        
        ste = calculate_ste(frames)
        log_ste = calculate_log_ste(ste)
        spectral_centroid = calculate_spectral_centroid(frames, fs)
        
        intervals = read_lab(lab_path)
        gt_boundaries = get_speech_boundaries(intervals)
        
        training_data.append({
            'log_ste': log_ste,
            'spectral_centroid': spectral_centroid,
            'feature_times': feature_times,
            'gt_boundaries': gt_boundaries,
            'filename': wav_path,
        })
    
    print(f"  Pre-computed features for {len(training_data)} files.\n")
    
    # Grid search
    best_params = None
    best_mae = float('inf')
    best_rmse = 0.0
    total_combos = (len(CANDIDATE_BINS) * len(CANDIDATE_SIGMA) *
                    len(CANDIDATE_W) * len(CANDIDATE_PROMINENCE) *
                    len(CANDIDATE_DISTANCE))
    
    print(f"Total parameter combinations: {total_combos}")
    print("Searching...")
    
    combo_count = 0
    for bins, sigma, W, prom, dist in itertools.product(
        CANDIDATE_BINS, CANDIDATE_SIGMA, CANDIDATE_W,
        CANDIDATE_PROMINENCE, CANDIDATE_DISTANCE
    ):
        combo_count += 1
        all_errors = []
        valid = True
        
        for data in training_data:
            try:
                # Compute histogram thresholds
                energy_diag = compute_histogram_threshold(
                    data['log_ste'], bins, sigma, W, prom, dist,
                    feature_name='energy'
                )
                centroid_diag = compute_histogram_threshold(
                    data['spectral_centroid'], bins, sigma, W, prom, dist,
                    feature_name='centroid'
                )
                
                # Check if thresholds were found
                if (energy_diag['threshold'] is None or 
                    centroid_diag['threshold'] is None):
                    valid = False
                    break
                
                # Classify frames
                speech_mask = _combine_feature_decisions(
                    data['log_ste'], energy_diag['threshold'],
                    data['spectral_centroid'], centroid_diag['threshold']
                )
                
                # Post-process
                speech_mask = merge_short_silences(
                    speech_mask, data['feature_times']
                )
                
                # Get boundaries
                segments = mask_to_segments(speech_mask, data['feature_times'])
                boundaries_s = segments_to_boundaries(segments)
                
                # Evaluate
                if len(boundaries_s) > 0 and len(data['gt_boundaries']) > 0:
                    eval_result = evaluate_boundaries(
                        boundaries_s, data['gt_boundaries']
                    )
                    all_errors.extend(eval_result['errors_ms'])
                    
            except Exception:
                valid = False
                break
        
        if not valid or len(all_errors) == 0:
            continue
        
        # Compute average MAE for this configuration
        avg_mae = float(np.mean(all_errors))
        avg_rmse = float(np.sqrt(np.mean(np.array(all_errors) ** 2)))
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_rmse = avg_rmse
            best_params = {
                'histogram_bins': bins,
                'smoothing_sigma': sigma,
                'W': W,
                'peak_prominence': prom,
                'peak_distance': dist,
                'training_mae_ms': round(best_mae, 2),
                'training_rmse_ms': round(best_rmse, 2),
                'feature_representation': 'log_STE + spectral_centroid',
            }
    
    if best_params is None:
        print("\n  ERROR: No valid parameter combination found!")
        print("  Using default parameters as fallback.")
        best_params = {
            'histogram_bins': 50,
            'smoothing_sigma': 3.0,
            'W': 5.0,
            'peak_prominence': 0.01,
            'peak_distance': 5,
            'training_mae_ms': -1.0,
            'training_rmse_ms': -1.0,
            'feature_representation': 'log_STE + spectral_centroid',
        }
    
    # Print results
    print(f"\n  Selected parameters:")
    print(f"    bins       = {best_params['histogram_bins']}")
    print(f"    sigma      = {best_params['smoothing_sigma']}")
    print(f"    W          = {best_params['W']}")
    print(f"    prominence = {best_params['peak_prominence']}")
    print(f"    distance   = {best_params['peak_distance']}")
    print(f"")
    print(f"  Training MAE  = {best_params['training_mae_ms']:.2f} ms")
    print(f"  Training RMSE = {best_params['training_rmse_ms']:.2f} ms")
    
    return best_params


if __name__ == '__main__':
    """Allow standalone Histogram training for debugging.
    
    Usage:
        python -m src.histogram.train
    """
    import os
    import glob
    from config.parameters import TRAINING_DIR, save_trained_parameters
    
    print("Histogram Training (standalone mode)")
    print("=" * 50)
    
    # Find training files
    wav_files = sorted(glob.glob(os.path.join(TRAINING_DIR, '*.wav')))
    lab_files = [os.path.splitext(f)[0] + '.lab' for f in wav_files]
    
    if not wav_files:
        print(f"\nERROR: No training WAV files found.")
        print(f"Please place the lecturer's training files in:")
        print(f"  {TRAINING_DIR}")
        exit(1)
    
    # Validate matching LAB files
    for lab_path in lab_files:
        if not os.path.exists(lab_path):
            print(f"\nERROR: Missing LAB file: {lab_path}")
            exit(1)
    
    print(f"Found {len(wav_files)} training files.")
    
    # Train
    params = train_histogram(wav_files, lab_files)
    
    # Save
    all_params = {'histogram': params}
    save_trained_parameters(all_params)
    
    print("\nHistogram training complete.")
