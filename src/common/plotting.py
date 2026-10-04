"""
Visualization utilities for speech/silence segmentation results.

Provides:
    - Common waveform + boundary plots for all algorithms
    - Histogram-specific diagnostic plots
    - Generic diagnostic subplot support

Conventions:
    - Ground-truth boundaries: RED vertical lines
    - Detected boundaries: BLUE vertical lines
    - All figures include title, axis labels, and legends
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt


def plot_algorithm_result(signal, fs, result, gt_boundaries_s,
                          output_path, test_filename):
    """Generate the main result figure for any algorithm.
    
    Dispatches to algorithm-specific plotting based on the algorithm name.
    
    For Histogram: generates a detailed 4-subplot figure.
    For other algorithms: generates a 3-subplot figure.
    
    Args:
        signal (np.ndarray): Original audio signal.
        fs (int): Sampling rate.
        result (dict): Algorithm result dictionary.
        gt_boundaries_s (list of float): Ground-truth boundaries in seconds.
        output_path (str): Path to save the figure.
        test_filename (str): Name of the test WAV file.
    """
    algorithm_name = result['algorithm_name']
    diagnostics = result.get('diagnostics', {})
    
    if algorithm_name == 'Histogram':
        _plot_histogram_result(signal, fs, result, gt_boundaries_s,
                               output_path, test_filename)
    else:
        _plot_generic_result(signal, fs, result, gt_boundaries_s,
                              output_path, test_filename)


def _plot_generic_result(signal, fs, result, gt_boundaries_s,
                          output_path, test_filename):
    """Generate a generic result figure for Binary Search or Simple Statistics.
    
    Layout:
        Plot 1: Waveform with boundaries
        Plot 2: Feature sequence with threshold
        Plot 3: Speech mask
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{result['algorithm_name']} — {test_filename}",
                 fontsize=14, fontweight='bold')
    
    # Time axis for waveform
    time_axis = np.arange(len(signal)) / fs
    
    # --- Plot 1: Waveform ---
    ax = axes[0]
    ax.plot(time_axis, signal, color='gray', linewidth=0.5, alpha=0.7)
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform with Speech Boundaries')
    _add_boundaries(ax, gt_boundaries_s, result.get('boundaries_ms', []))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: Feature ---
    ax = axes[1]
    feature_times = result.get('feature_times', np.array([]))
    features = result.get('features', {})
    thresholds = result.get('thresholds', {})
    
    # Plot first available feature
    for feat_name, feat_values in features.items():
        if feat_values is not None and len(feat_values) > 0:
            ax.plot(feature_times, feat_values, linewidth=0.8, label=feat_name)
    
    # Plot thresholds
    for thr_name, thr_value in thresholds.items():
        if thr_value is not None:
            ax.axhline(y=thr_value, color='red', linestyle='--',
                      linewidth=1.5, label=f'Threshold: {thr_value:.4f}')
    
    ax.set_ylabel('Feature Value')
    ax.set_title('Feature Sequence')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 3: Speech Mask ---
    ax = axes[2]
    speech_mask = result.get('speech_mask', np.array([]))
    if len(speech_mask) > 0 and len(feature_times) > 0:
        ax.fill_between(feature_times, speech_mask, alpha=0.4,
                       color='green', label='Detected Speech')
    ax.set_ylabel('Speech (1) / Silence (0)')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Speech Mask (after post-processing)')
    ax.set_ylim(-0.1, 1.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_histogram_result(signal, fs, result, gt_boundaries_s,
                            output_path, test_filename):
    """Generate a detailed Histogram result figure.
    
    Layout:
        Plot 1: Waveform with ground-truth (RED) and detected (BLUE) boundaries
        Plot 2: Energy and Spectral Centroid feature sequences with thresholds
        Plot 3: Energy histogram diagnostics (histogram, smoothed, M1, M2, threshold)
        Plot 4: Spectral Centroid histogram diagnostics
    """
    diagnostics = result.get('diagnostics', {})
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f"Histogram Algorithm — {test_filename}",
                 fontsize=14, fontweight='bold')
    
    # Time axis for waveform
    time_axis = np.arange(len(signal)) / fs
    feature_times = result.get('feature_times', np.array([]))
    
    # ========================================
    # Plot 1: Waveform with boundaries
    # ========================================
    ax = axes[0]
    ax.plot(time_axis, signal, color='gray', linewidth=0.5, alpha=0.7)
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform with Speech Boundaries')
    _add_boundaries(ax, gt_boundaries_s, result.get('boundaries_ms', []))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (seconds)')
    
    # ========================================
    # Plot 2: Feature sequences with thresholds
    # ========================================
    ax1 = axes[1]
    
    energy_feature = diagnostics.get('energy_feature', np.array([]))
    spectral_centroid = diagnostics.get('spectral_centroid', np.array([]))
    energy_threshold = diagnostics.get('energy_threshold')
    centroid_threshold = diagnostics.get('centroid_threshold')
    
    if len(energy_feature) > 0:
        color_e = 'tab:blue'
        ax1.plot(feature_times, energy_feature, color=color_e,
                linewidth=0.8, label='Energy (log STE)', alpha=0.8)
        if energy_threshold is not None:
            ax1.axhline(y=energy_threshold, color=color_e, linestyle='--',
                       linewidth=1.5, label=f'Energy Threshold: {energy_threshold:.4f}')
    
    ax1.set_ylabel('Energy (log STE)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Second y-axis for spectral centroid
    if len(spectral_centroid) > 0:
        ax2 = ax1.twinx()
        color_c = 'tab:orange'
        ax2.plot(feature_times, spectral_centroid, color=color_c,
                linewidth=0.8, label='Spectral Centroid', alpha=0.8)
        if centroid_threshold is not None:
            ax2.axhline(y=centroid_threshold, color=color_c, linestyle='--',
                       linewidth=1.5, label=f'Centroid Threshold: {centroid_threshold:.1f} Hz')
        ax2.set_ylabel('Spectral Centroid (Hz)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
    else:
        ax1.legend(loc='upper right', fontsize=8)
    
    ax1.set_title('Feature Sequences with Thresholds')
    ax1.set_xlabel('Time (seconds)')
    ax1.grid(True, alpha=0.3)
    
    # ========================================
    # Plot 3: Energy Histogram Diagnostics
    # ========================================
    ax = axes[2]
    _plot_histogram_diagnostic(
        ax,
        diagnostics.get('energy_histogram'),
        diagnostics.get('energy_histogram_smoothed'),
        diagnostics.get('energy_histogram_bins'),
        diagnostics.get('energy_M1'),
        diagnostics.get('energy_M2'),
        diagnostics.get('energy_threshold'),
        title='Energy Histogram',
        xlabel='Energy (log STE)'
    )
    
    # ========================================
    # Plot 4: Spectral Centroid Histogram Diagnostics
    # ========================================
    ax = axes[3]
    _plot_histogram_diagnostic(
        ax,
        diagnostics.get('centroid_histogram'),
        diagnostics.get('centroid_histogram_smoothed'),
        diagnostics.get('centroid_histogram_bins'),
        diagnostics.get('centroid_M1'),
        diagnostics.get('centroid_M2'),
        diagnostics.get('centroid_threshold'),
        title='Spectral Centroid Histogram',
        xlabel='Spectral Centroid (Hz)'
    )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_histogram_diagnostic(ax, histogram, smoothed, bin_edges,
                                M1, M2, threshold, title, xlabel):
    """Plot a single histogram diagnostic subplot.
    
    Shows:
        - Original histogram (gray bars)
        - Smoothed histogram (blue line)
        - M1 (green vertical line)
        - M2 (orange vertical line)
        - Threshold (red dashed line)
    """
    if histogram is None or bin_edges is None:
        ax.text(0.5, 0.5, 'No histogram data available',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_title(title)
        return
    
    # Plot histogram bars
    bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2
    bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1
    
    ax.bar(bin_centers, histogram, width=bin_width * 0.8,
          color='lightgray', edgecolor='gray', alpha=0.7, label='Histogram')
    
    # Plot smoothed histogram
    if smoothed is not None:
        ax.plot(bin_centers, smoothed, color='tab:blue', linewidth=2,
               label='Smoothed')
    
    # Plot M1
    if M1 is not None:
        ax.axvline(x=M1, color='green', linewidth=2,
                  linestyle='-', label=f'M1 = {M1:.4f}')
    
    # Plot M2
    if M2 is not None:
        ax.axvline(x=M2, color='orange', linewidth=2,
                  linestyle='-', label=f'M2 = {M2:.4f}')
    
    # Plot threshold
    if threshold is not None:
        ax.axvline(x=threshold, color='red', linewidth=2,
                  linestyle='--', label=f'Threshold = {threshold:.4f}')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def _add_boundaries(ax, gt_boundaries_s, detected_boundaries_ms):
    """Add ground-truth and detected boundary lines to an axis.
    
    Conventions:
        Ground-truth: RED vertical lines
        Detected: BLUE vertical lines
    
    Args:
        ax: Matplotlib axis.
        gt_boundaries_s (list of float): Ground-truth boundaries in seconds.
        detected_boundaries_ms (list of float): Detected boundaries in
            milliseconds.
    """
    gt_plotted = False
    det_plotted = False
    
    for b in gt_boundaries_s:
        label = 'Ground Truth' if not gt_plotted else None
        ax.axvline(x=b, color='red', linewidth=1.5, linestyle='-',
                  alpha=0.8, label=label)
        gt_plotted = True
    
    for b_ms in detected_boundaries_ms:
        b_s = b_ms / 1000.0
        label = 'Detected' if not det_plotted else None
        ax.axvline(x=b_s, color='blue', linewidth=1.5, linestyle='--',
                  alpha=0.8, label=label)
        det_plotted = True


def plot_snr_comparison(snr_results, output_path):
    """Generate SNR experiment comparison plot.
    
    Shows MAE and RMSE vs SNR for each algorithm.
    
    Args:
        snr_results (list of dict): Each dict has:
            'algorithm' (str): Algorithm name
            'snr_db' (float): SNR level
            'mae_ms' (float): MAE
            'rmse_ms' (float): RMSE
        output_path (str): Path to save the figure.
    """
    import pandas as pd
    
    if not snr_results:
        return
    
    df = pd.DataFrame(snr_results)
    algorithms = df['algorithm'].unique()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('SNR Robustness Experiment', fontsize=14, fontweight='bold')
    
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo].sort_values('snr_db')
        ax1.plot(algo_data['snr_db'], algo_data['mae_ms'],
                marker='o', linewidth=2, label=algo)
        ax2.plot(algo_data['snr_db'], algo_data['rmse_ms'],
                marker='s', linewidth=2, label=algo)
    
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('MAE (ms)')
    ax1.set_title('MAE vs SNR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Higher SNR on left
    
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('RMSE (ms)')
    ax2.set_title('RMSE vs SNR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
