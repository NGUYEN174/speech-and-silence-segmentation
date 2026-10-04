"""
Post-processing utilities for speech/silence segmentation.

Provides short-silence merging: if a silence gap between two speech
regions is shorter than MIN_SILENCE_MS (200 ms, fixed by the lecturer),
the neighboring speech regions are merged.

This parameter (200 ms) MUST NOT be optimized.
"""

import numpy as np
from config.parameters import MIN_SILENCE_MS


def merge_short_silences(mask, frame_times, min_silence_ms=None):
    """Merge speech regions separated by short silence gaps.
    
    If a silence gap between two speech regions is shorter than
    min_silence_ms, the silence frames are converted to speech,
    effectively merging the two speech regions.
    
    Example:
        Before: Speech | Silence (150ms) | Speech
        After:  Speech | Speech  (merged) | Speech
        
        Before: Speech | Silence (300ms) | Speech  
        After:  Speech | Silence (300ms) | Speech   (unchanged)
    
    Args:
        mask (np.ndarray): Binary array where 1 = speech, 0 = silence.
        frame_times (np.ndarray): Time (in seconds) for each frame.
        min_silence_ms (float, optional): Minimum silence duration in ms.
            Defaults to MIN_SILENCE_MS (200 ms, fixed by lecturer).
    
    Returns:
        np.ndarray: Updated binary mask with short silences merged.
    """
    if min_silence_ms is None:
        min_silence_ms = MIN_SILENCE_MS
    
    min_silence_s = min_silence_ms / 1000.0
    
    # Work on a copy
    merged = mask.copy()
    
    if len(merged) == 0:
        return merged
    
    # Find silence gaps
    padded = np.concatenate([[1], merged, [1]])
    diff = np.diff(padded)
    
    # Start of silence (1 -> 0)
    sil_starts = np.where(diff == -1)[0]
    # End of silence (0 -> 1)
    sil_ends = np.where(diff == 1)[0]
    
    for s, e in zip(sil_starts, sil_ends):
        # s is the first silence frame index
        # e is the first speech frame after the silence
        # Silence duration
        s_clamped = min(s, len(frame_times) - 1)
        e_clamped = min(e - 1, len(frame_times) - 1)
        
        if s_clamped >= len(frame_times) or e_clamped < 0:
            continue
        
        frame_shift_s = np.median(np.diff(frame_times))
        silence_duration = (e - s) * frame_shift_s
        
        if silence_duration < min_silence_s:
            # Merge: set all silence frames to speech
            merged[s:e] = 1
    
    return merged
