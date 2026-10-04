"""
Audio feature extraction functions.

Provides:
    - Short-Time Energy (STE)
    - Log Short-Time Energy
    - Spectral Centroid

All feature functions operate on pre-framed signals.
The same feature representation MUST be used for both training and testing.

IMPORTANT:
    These are reusable common functions. Algorithm-specific feature
    processing belongs in the respective algorithm module.
"""

import numpy as np
from config.parameters import EPSILON


def calculate_ste(frames):
    """Calculate Short-Time Energy for each frame.
    
    Formula:
        STE_i = (1/N) * sum(x_i[n]^2)  for n = 0..N-1
    
    where:
        N = frame length in samples
        x_i[n] = samples of frame i
    
    This is the MEAN squared amplitude (normalized by frame length).
    Normalization makes the energy comparable across different frame sizes.
    
    Args:
        frames (np.ndarray): 2D array of shape (num_frames, frame_length).
            Each row is one frame of audio samples.
    
    Returns:
        np.ndarray: 1D array of STE values, one per frame.
    """
    if frames.shape[0] == 0:
        return np.array([])
    
    # STE = mean of squared samples per frame
    ste = np.mean(frames ** 2, axis=1)
    return ste


def calculate_log_ste(ste):
    """Calculate log10 of Short-Time Energy.
    
    Formula:
        log_STE_i = log10(STE_i + epsilon)
    
    where:
        epsilon = 1e-12 (small constant to avoid log(0))
    
    Adding epsilon prevents -inf values for silent frames.
    
    Args:
        ste (np.ndarray): 1D array of STE values.
    
    Returns:
        np.ndarray: 1D array of log10(STE + epsilon) values.
    """
    return np.log10(ste + EPSILON)


def calculate_spectral_centroid(frames, fs):
    """Calculate Spectral Centroid for each frame.
    
    The spectral centroid is the "center of mass" of the spectrum.
    It indicates where the "center" of the spectral energy is located.
    
    Formula:
        C_i = sum(f_k * |X_i(k)|) / sum(|X_i(k)|)
    
    where:
        f_k = frequency of bin k (in Hz)
        |X_i(k)| = magnitude spectrum of frame i at bin k
        The sum is over k = 0, 1, ..., N/2 (positive frequencies only)
    
    FFT/Windowing convention:
        - No explicit window is applied here (rectangular window).
          If windowing is desired, apply it to frames before calling.
        - FFT size = frame length (no zero-padding beyond frame length).
        - Only positive frequencies (0 to fs/2) are used.
    
    Magnitude spectrum:
        |X(k)| = absolute value of the complex FFT coefficients
        for the positive frequency bins.
    
    Frequency bin calculation:
        f_k = k * fs / N,  for k = 0, 1, ..., N//2
        where N is the FFT size (= frame length).
    
    Zero-denominator handling:
        If a frame has zero total magnitude (completely silent frame),
        the spectral centroid is set to 0.0 Hz to avoid division by zero.
    
    Args:
        frames (np.ndarray): 2D array of shape (num_frames, frame_length).
        fs (int): Sampling rate in Hz.
    
    Returns:
        np.ndarray: 1D array of spectral centroid values (in Hz),
            one per frame.
    """
    if frames.shape[0] == 0:
        return np.array([])
    
    num_frames, frame_length = frames.shape
    
    # Compute FFT for all frames
    # np.fft.rfft computes the FFT for real-valued input,
    # returning only positive frequencies (0 to fs/2)
    fft_result = np.fft.rfft(frames, axis=1)
    
    # Magnitude spectrum: |X(k)|
    magnitudes = np.abs(fft_result)
    
    # Frequency bins: f_k = k * fs / N
    # rfft returns N//2 + 1 frequency bins
    num_bins = magnitudes.shape[1]
    frequencies = np.arange(num_bins) * fs / frame_length
    
    # Spectral centroid: C = sum(f * |X|) / sum(|X|)
    # Compute for all frames at once using matrix operations
    weighted_sum = np.dot(magnitudes, frequencies)    # sum(f_k * |X_k|)
    total_magnitude = np.sum(magnitudes, axis=1)       # sum(|X_k|)
    
    # Handle zero-denominator: set centroid to 0.0 for silent frames
    centroid = np.zeros(num_frames)
    nonzero_mask = total_magnitude > 0
    centroid[nonzero_mask] = (
        weighted_sum[nonzero_mask] / total_magnitude[nonzero_mask]
    )
    
    return centroid
