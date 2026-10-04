"""
Signal framing utilities.

Divides a continuous audio signal into overlapping frames.

Fixed parameters (set by the lecturer - NOT tunable):
    Frame length: 25 ms
    Frame shift:  10 ms

Frame length and shift in samples are computed from the actual
sampling rate of each WAV file:
    frame_length_samples = round(0.025 * fs)
    frame_shift_samples  = round(0.010 * fs)

Incomplete final frame handling:
    Strategy: DISCARD incomplete final frames.
    
    Rationale: Discarding avoids introducing zero-padded artifacts
    that could affect energy and spectral centroid calculations.
    This strategy is used consistently for both training and testing.
"""

import numpy as np
from config.parameters import FRAME_LENGTH_MS, FRAME_SHIFT_MS


def frame_signal(signal, fs, frame_ms=None, hop_ms=None):
    """Divide a signal into overlapping frames.
    
    Computes frame length and hop size in samples from the actual
    sampling rate. Does NOT hard-code sample counts.
    
    Incomplete final frame handling:
        If the remaining samples at the end of the signal are fewer
        than one full frame, that incomplete frame is DISCARDED.
        This is consistent for both training and testing.
    
    Args:
        signal (np.ndarray): 1D audio signal array.
        fs (int): Sampling rate in Hz.
        frame_ms (float, optional): Frame length in ms. 
            Defaults to FRAME_LENGTH_MS (25 ms).
        hop_ms (float, optional): Frame shift in ms.
            Defaults to FRAME_SHIFT_MS (10 ms).
    
    Returns:
        dict with keys:
            'frames' (np.ndarray): 2D array of shape (num_frames, frame_length)
                Each row is one frame.
            'start_samples' (np.ndarray): Start sample index of each frame.
            'start_times' (np.ndarray): Start time in seconds of each frame.
            'center_times' (np.ndarray): Center time in seconds of each frame.
            'frame_length_samples' (int): Frame length in samples.
            'hop_samples' (int): Hop size in samples.
            'num_frames' (int): Total number of frames.
    """
    if frame_ms is None:
        frame_ms = FRAME_LENGTH_MS
    if hop_ms is None:
        hop_ms = FRAME_SHIFT_MS
    
    # Compute frame parameters from actual sampling rate
    frame_length = round(frame_ms / 1000.0 * fs)
    hop_length = round(hop_ms / 1000.0 * fs)
    
    # Calculate number of complete frames
    # A frame starting at index i needs samples i to i + frame_length - 1
    # The last valid start index is len(signal) - frame_length
    num_samples = len(signal)
    if num_samples < frame_length:
        # Signal is shorter than one frame
        return {
            'frames': np.empty((0, frame_length)),
            'start_samples': np.array([], dtype=int),
            'start_times': np.array([]),
            'center_times': np.array([]),
            'frame_length_samples': frame_length,
            'hop_samples': hop_length,
            'num_frames': 0,
        }
    
    # Start indices for each frame
    start_samples = np.arange(0, num_samples - frame_length + 1, hop_length)
    num_frames = len(start_samples)
    
    # Extract frames using array indexing
    # Each frame is signal[start : start + frame_length]
    indices = start_samples[:, np.newaxis] + np.arange(frame_length)
    frames = signal[indices]
    
    # Compute timing information
    start_times = start_samples / fs
    center_times = (start_samples + frame_length / 2.0) / fs
    
    return {
        'frames': frames,
        'start_samples': start_samples,
        'start_times': start_times,
        'center_times': center_times,
        'frame_length_samples': frame_length,
        'hop_samples': hop_length,
        'num_frames': num_frames,
    }
