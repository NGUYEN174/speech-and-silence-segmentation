"""
Main demonstration script for Speech/Silence Segmentation.

Usage:
    python run_all.py                    # Run ALL algorithms
    python run_all.py --algorithm histogram   # Histogram only (development)
    python run_all.py --snr              # Also run SNR experiment

This script:
    1. Loads trained parameters from config/trained_parameters.json
    2. Scans test directory for WAV/LAB files
    3. Runs each algorithm on all test files
    4. Evaluates results against ground truth
    5. Generates result figures (12 figures for 3 algorithms x 4 files)
    6. Creates summary.csv
    7. Prints final comparison table

IMPORTANT:
    - Test LAB files are used ONLY for evaluation, NEVER for tuning.
    - Parameters are frozen (loaded from trained_parameters.json).
    - The student should press Run once during demonstration.
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

from config.parameters import (
    TESTING_DIR, FRAME_LENGTH_MS, FRAME_SHIFT_MS, MIN_SILENCE_MS,
    RESULTS_DIR, RESULTS_SNR, SNR_LEVELS_DB, SNR_RANDOM_SEED,
    load_trained_parameters, ensure_results_dirs
)
from src.common.audio import load_audio
from src.common.lab_reader import read_lab, get_speech_boundaries
from src.common.evaluation import evaluate_boundaries
from src.common.plotting import plot_algorithm_result, plot_snr_comparison
from src.common.interfaces import get_all_algorithms


def find_test_pairs(testing_dir):
    """Find all WAV/LAB file pairs in the test directory.
    
    Args:
        testing_dir (str): Path to test data directory.
    
    Returns:
        tuple: (wav_files, lab_files) - sorted lists of matching paths.
    """
    if not os.path.exists(testing_dir):
        print(f"\nERROR: Test directory not found:")
        print(f"  {testing_dir}")
        print(f"\nPlease create the directory and place the lecturer's")
        print(f"test files (*.wav and *.lab) there.")
        sys.exit(1)
    
    wav_files = sorted(glob.glob(os.path.join(testing_dir, '*.wav')))
    
    if not wav_files:
        print(f"\nERROR: No test WAV files found.")
        print(f"\nPlease place the lecturer's test files in:")
        print(f"  {testing_dir}")
        sys.exit(1)
    
    # Validate expected count
    if len(wav_files) != 4:
        print(f"\n  WARNING: Expected 4 test WAV files, found {len(wav_files)}.")
        print(f"  Proceeding with {len(wav_files)} files.")
    
    lab_files = []
    for wav_path in wav_files:
        lab_path = os.path.splitext(wav_path)[0] + '.lab'
        if not os.path.exists(lab_path):
            print(f"\nERROR: Missing LAB file for {os.path.basename(wav_path)}")
            print(f"  Expected: {lab_path}")
            sys.exit(1)
        lab_files.append(lab_path)
    
    return wav_files, lab_files


def add_noise_at_snr(signal, snr_db, rng):
    """Add white Gaussian noise to a signal at a specified SNR.
    
    Does NOT modify the original signal.
    
    Args:
        signal (np.ndarray): Clean audio signal.
        snr_db (float): Desired Signal-to-Noise Ratio in dB.
        rng (np.random.Generator): Random number generator.
    
    Returns:
        np.ndarray: Noisy signal.
    """
    signal_power = np.mean(signal ** 2)
    # SNR = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / (10 ** (snr_db / 10))
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    noise = rng.normal(0, noise_std, len(signal))
    return signal + noise


def run_snr_experiment(algorithms, trained_params, wav_files, lab_files):
    """Run the SNR robustness experiment.
    
    For each algorithm and each SNR level:
        1. Add noise to each test file at the specified SNR
        2. Run the algorithm on the noisy signal
        3. Evaluate against ground truth
    
    Args:
        algorithms (list): Algorithm instances.
        trained_params (dict): Trained parameters.
        wav_files (list): Test WAV file paths.
        lab_files (list): Test LAB file paths.
    
    Returns:
        list of dict: SNR experiment results.
    """
    print(f"\n{'=' * 60}")
    print(f"  SNR ROBUSTNESS EXPERIMENT")
    print(f"{'=' * 60}")
    print(f"  SNR levels: {SNR_LEVELS_DB} dB")
    print(f"  Random seed: {SNR_RANDOM_SEED}")
    
    rng = np.random.default_rng(SNR_RANDOM_SEED)
    snr_results = []
    
    for algo in algorithms:
        if algo.is_placeholder:
            print(f"\n  [{algo.name}] Skipping (placeholder).")
            continue
        
        params = trained_params.get(algo.key, {})
        
        for snr_db in SNR_LEVELS_DB:
            all_errors = []
            
            for wav_path, lab_path in zip(wav_files, lab_files):
                signal, fs = load_audio(wav_path)
                
                # Add noise
                noisy_signal = add_noise_at_snr(signal, snr_db, rng)
                
                # Predict
                try:
                    result = algo.predict(noisy_signal, fs, params)
                    if result is None:
                        continue
                    
                    # Evaluate
                    intervals = read_lab(lab_path)
                    gt_boundaries = get_speech_boundaries(intervals)
                    detected_boundaries_s = [
                        b / 1000.0 for b in result['boundaries_ms']
                    ]
                    
                    eval_result = evaluate_boundaries(
                        detected_boundaries_s, gt_boundaries
                    )
                    all_errors.extend(eval_result['errors_ms'])
                except Exception as e:
                    print(f"    Error with {os.path.basename(wav_path)} "
                          f"at SNR={snr_db}dB: {e}")
            
            if all_errors:
                avg_mae = float(np.mean(all_errors))
                avg_rmse = float(np.sqrt(np.mean(np.array(all_errors) ** 2)))
            else:
                avg_mae = float('nan')
                avg_rmse = float('nan')
            
            snr_results.append({
                'algorithm': algo.name,
                'snr_db': snr_db,
                'mae_ms': round(avg_mae, 2),
                'rmse_ms': round(avg_rmse, 2),
            })
            
            print(f"  [{algo.name}] SNR={snr_db:3d} dB: "
                  f"MAE={avg_mae:.2f} ms, RMSE={avg_rmse:.2f} ms")
    
    # Save SNR results
    if snr_results:
        snr_df = pd.DataFrame(snr_results)
        snr_csv_path = os.path.join(RESULTS_SNR, 'snr_results.csv')
        os.makedirs(RESULTS_SNR, exist_ok=True)
        snr_df.to_csv(snr_csv_path, index=False)
        print(f"\n  SNR results saved to: {snr_csv_path}")
        
        # Generate SNR plot
        snr_plot_path = os.path.join(RESULTS_SNR, 'snr_comparison.png')
        plot_snr_comparison(snr_results, snr_plot_path)
        print(f"  SNR plot saved to: {snr_plot_path}")
    
    return snr_results


def main():
    """Main demonstration pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Speech/Silence Segmentation - Test & Evaluation'
    )
    parser.add_argument(
        '--algorithm', type=str, default=None,
        help='Run only a specific algorithm (e.g., histogram). '
             'Default: run all algorithms.'
    )
    parser.add_argument(
        '--snr', action='store_true',
        help='Also run SNR robustness experiment.'
    )
    args = parser.parse_args()
    
    # Header
    print("=" * 60)
    print("  SPEECH/SILENCE SEGMENTATION - TEST & EVALUATION")
    print("=" * 60)
    
    # Print fixed parameters
    print(f"\nFixed parameters:")
    print(f"  Frame length : {FRAME_LENGTH_MS} ms")
    print(f"  Frame shift  : {FRAME_SHIFT_MS} ms")
    print(f"  Min silence  : {MIN_SILENCE_MS} ms")
    
    # Load trained parameters
    print(f"\nLoading trained parameters...")
    try:
        trained_params = load_trained_parameters()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    print("  Parameters loaded successfully.")
    
    # Find test files
    print(f"\nSearching for test files in:")
    print(f"  {TESTING_DIR}")
    wav_files, lab_files = find_test_pairs(TESTING_DIR)
    print(f"  Found {len(wav_files)} test files.")
    for wav in wav_files:
        print(f"    {os.path.basename(wav)}")
    
    # Ensure results directories exist
    ensure_results_dirs()
    
    # Get algorithms
    all_algorithms = get_all_algorithms()
    
    # Filter by algorithm if specified
    if args.algorithm:
        all_algorithms = [
            a for a in all_algorithms
            if a.key == args.algorithm or a.name.lower() == args.algorithm.lower()
        ]
        if not all_algorithms:
            print(f"\nERROR: Algorithm '{args.algorithm}' not found.")
            print(f"Available algorithms: "
                  f"{[a.key for a in get_all_algorithms()]}")
            sys.exit(1)
        print(f"\n  Running only: {all_algorithms[0].name}")
    
    # ======================================
    # Run each algorithm on all test files
    # ======================================
    all_results = []  # For summary.csv
    
    for algo in all_algorithms:
        print(f"\n{'=' * 60}")
        print(f"  ALGORITHM: {algo.name}")
        print(f"{'=' * 60}")
        
        if algo.is_placeholder:
            print(f"\n  Status: PLACEHOLDER (not yet implemented)")
            print(f"  Skipping evaluation. No fake results generated.")
            continue
        
        params = trained_params.get(algo.key, {})
        
        for wav_path, lab_path in zip(wav_files, lab_files):
            test_filename = os.path.basename(wav_path)
            test_name = os.path.splitext(test_filename)[0]
            
            print(f"\n  [{algo.name}] {test_filename}")
            print(f"  {'-' * 40}")
            
            # Load audio
            signal, fs = load_audio(wav_path)
            print(f"    Sampling rate: {fs} Hz")
            print(f"    Duration: {len(signal)/fs:.2f} seconds")
            
            # Predict
            result = algo.predict(signal, fs, params)
            
            if result is None:
                print(f"    No result (algorithm returned None).")
                continue
            
            # Load ground truth
            intervals = read_lab(lab_path)
            gt_boundaries_s = get_speech_boundaries(intervals)
            gt_boundaries_ms = [b * 1000.0 for b in gt_boundaries_s]
            
            # Print algorithm-specific diagnostics
            if algo.key == 'histogram':
                diag = result.get('diagnostics', {})
                print(f"    Energy threshold: {diag.get('energy_threshold', 'N/A')}")
                print(f"    Centroid threshold: "
                      f"{diag.get('centroid_threshold', 'N/A')}")
            
            # Print segments
            segments = result.get('segments', [])
            detected_boundaries_ms = result.get('boundaries_ms', [])
            print(f"    Detected segments: {len(segments)}")
            print(f"    Detected boundaries (ms): "
                  f"{[round(b, 1) for b in detected_boundaries_ms]}")
            print(f"    Ground truth boundaries (ms): "
                  f"{[round(b, 1) for b in gt_boundaries_ms]}")
            
            # Evaluate
            detected_boundaries_s = [b / 1000.0 for b in detected_boundaries_ms]
            eval_result = evaluate_boundaries(
                detected_boundaries_s, gt_boundaries_s
            )
            
            print(f"    MAE:  {eval_result['mae_ms']:.2f} ms")
            print(f"    RMSE: {eval_result['rmse_ms']:.2f} ms")
            print(f"    Matched: {eval_result['num_matched']}, "
                  f"Missing: {eval_result['num_missing']}, "
                  f"Extra: {eval_result['num_extra']}")
            
            # Store for summary
            all_results.append({
                'Algorithm': algo.name,
                'TestFile': test_filename,
                'MAE_ms': round(eval_result['mae_ms'], 2),
                'RMSE_ms': round(eval_result['rmse_ms'], 2),
                'NumDetectedBoundaries': eval_result['num_det'],
                'NumGroundTruthBoundaries': eval_result['num_gt'],
                'NumMatched': eval_result['num_matched'],
                'NumMissing': eval_result['num_missing'],
                'NumExtra': eval_result['num_extra'],
                'BoundaryErrors_ms': str([round(e, 2) for e in eval_result['errors_ms']]),
            })
            
            # Generate figure
            results_subdir = os.path.join(RESULTS_DIR, algo.key)
            os.makedirs(results_subdir, exist_ok=True)
            figure_path = os.path.join(results_subdir, f"{test_name}.png")
            
            plot_algorithm_result(
                signal, fs, result, gt_boundaries_s,
                figure_path, test_filename
            )
            print(f"    Figure saved: {figure_path}")
    
    # ======================================
    # Generate summary.csv
    # ======================================
    if all_results:
        print(f"\n{'=' * 60}")
        print(f"  GENERATING SUMMARY")
        print(f"{'=' * 60}")
        
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(RESULTS_DIR, 'summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved to: {summary_path}")
        
        # ======================================
        # Final comparison table
        # ======================================
        print(f"\n{'=' * 60}")
        print(f"  FINAL COMPARISON")
        print(f"{'=' * 60}")
        
        print(f"\n  {'Algorithm':<25} {'Avg MAE':<15} {'Avg RMSE':<15}")
        print(f"  {'-' * 55}")
        
        best_algo = None
        best_mae = float('inf')
        
        for algo_name in df['Algorithm'].unique():
            algo_data = df[df['Algorithm'] == algo_name]
            avg_mae = algo_data['MAE_ms'].mean()
            avg_rmse = algo_data['RMSE_ms'].mean()
            
            print(f"  {algo_name:<25} {avg_mae:<15.2f} {avg_rmse:<15.2f}")
            
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_algo = algo_name
        
        print(f"\n  Lowest average MAE: {best_algo} ({best_mae:.2f} ms)")
    
    # ======================================
    # SNR experiment (if requested)
    # ======================================
    if args.snr:
        non_placeholder = [a for a in all_algorithms if not a.is_placeholder]
        if non_placeholder:
            run_snr_experiment(
                non_placeholder, trained_params, wav_files, lab_files
            )
    
    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
