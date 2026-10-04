"""
Segment and boundary conversion utilities.

Converts between:
    - Binary speech masks (per-frame)
    - Speech segments (list of start/end pairs)
    - Flat boundary lists
"""

import numpy as np


def mask_to_segments(mask, frame_times):
    """Convert a binary frame-level speech mask to speech segments.
    
    Finds contiguous runs of speech frames (mask == 1) and returns
    their start and end times.
    
    Args:
        mask (np.ndarray): Binary array where 1 = speech, 0 = silence.
        frame_times (np.ndarray): Time (in seconds) for each frame.
            Typically the center time or start time of each frame.
    
    Returns:
        list of dict: Each dict has:
            'start' (float): Start time of the speech segment (seconds)
            'end' (float): End time of the speech segment (seconds)
    """
    if len(mask) == 0:
        return []
    
    segments = []
    mask = np.array(mask, dtype=int)
    
    # Detect transitions by computing the difference
    # Prepend and append 0 to detect segments at the boundaries
    padded = np.concatenate([[0], mask, [0]])
    diff = np.diff(padded)
    
    # Rising edges (0 -> 1): start of speech
    starts = np.where(diff == 1)[0]
    # Falling edges (1 -> 0): end of speech  
    ends = np.where(diff == -1)[0]
    
    for s, e in zip(starts, ends):
        # s is the index of the first speech frame
        # e is the index AFTER the last speech frame
        # Clamp to valid range
        s = min(s, len(frame_times) - 1)
        e_idx = min(e - 1, len(frame_times) - 1)  # last speech frame index
        
        segments.append({
            'start': float(frame_times[s]),
            'end': float(frame_times[e_idx]),
        })
    
    return segments


def segments_to_boundaries(segments):
    """Convert speech segments to a flat boundary list.
    
    Args:
        segments (list of dict): Each dict has 'start' and 'end' keys.
    
    Returns:
        list of float: [start1, end1, start2, end2, ...] in seconds.
    """
    boundaries = []
    for seg in segments:
        boundaries.append(seg['start'])
        boundaries.append(seg['end'])
    return boundaries


def boundaries_to_ms(boundaries):
    """Convert boundary list from seconds to milliseconds.
    
    Args:
        boundaries (list of float): Boundaries in seconds.
    
    Returns:
        list of float: Boundaries in milliseconds.
    """
    return [b * 1000.0 for b in boundaries]
