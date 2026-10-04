
"""
Segment and boundary conversion utilities.

Converts between:
    - Binary speech masks (per-frame)
    - Speech segments (list of start/end pairs)
    - Flat boundary lists

Frame labels are assigned using frame center times.
Segment boundaries are estimated halfway between adjacent
frame centers where the classification changes.
"""

import numpy as np


def _estimate_frame_shift(frame_times):
    """Estimate the frame shift in seconds.

    The frame shift is estimated as the median difference
    between consecutive frame center times.

    If there is only one frame, use the project's fixed
    frame shift of 10 ms.
    """
    frame_times = np.asarray(frame_times, dtype=float)

    if len(frame_times) >= 2:
        differences = np.diff(frame_times)
        positive_differences = differences[differences > 0]

        if len(positive_differences) > 0:
            return float(np.median(positive_differences))

    return 0.010


def mask_to_segments(mask, frame_times):
    """Convert a binary speech mask to speech segments.

    Args:
        mask (np.ndarray):
            Binary array: 1 = speech, 0 = silence.

        frame_times (np.ndarray):
            Center time of each frame in seconds.

    Returns:
        list of dict:
            Each dictionary contains:
                'start': segment start time in seconds
                'end': segment end time in seconds

    Notes:
        Segment boundaries are estimated halfway between
        adjacent frame centers.

        A one-frame speech segment therefore has a positive
        duration instead of start == end.
    """
    mask = np.asarray(mask, dtype=int)
    frame_times = np.asarray(frame_times, dtype=float)

    if len(mask) == 0:
        return []

    if len(mask) != len(frame_times):
        raise ValueError(
            "mask and frame_times must have the same length."
        )

    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("mask must contain only 0 and 1.")

    if len(frame_times) > 1 and np.any(np.diff(frame_times) <= 0):
        raise ValueError("frame_times must be strictly increasing.")

    frame_shift = _estimate_frame_shift(frame_times)
    half_shift = frame_shift / 2.0

    # Detect transitions between silence and speech.
    padded = np.concatenate(([0], mask, [0]))
    diff = np.diff(padded)

    # Index of the first speech frame in each segment.
    starts = np.where(diff == 1)[0]

    # Index immediately after the last speech frame.
    ends = np.where(diff == -1)[0]

    segments = []

    for start_idx, end_idx in zip(starts, ends):
        first_center = frame_times[start_idx]
        last_center = frame_times[end_idx - 1]

        # Estimate boundaries halfway between frame centers.
        start_time = first_center - half_shift
        end_time = last_center + half_shift

        # Do not produce negative start times.
        start_time = max(0.0, start_time)

        # Guard against numerical or indexing edge cases.
        if end_time <= start_time:
            end_time = start_time + frame_shift

        segments.append({
            'start': float(start_time),
            'end': float(end_time),
        })

    return segments


def segments_to_boundaries(segments):
    """Convert segments to a flat list of boundaries.

    Args:
        segments (list of dict):
            Each dictionary contains 'start' and 'end'.

    Returns:
        list of float:
            [start1, end1, start2, end2, ...] in seconds.
    """
    boundaries = []

    for segment in segments:
        boundaries.append(float(segment['start']))
        boundaries.append(float(segment['end']))

    return boundaries


def boundaries_to_ms(boundaries):
    """Convert boundary times from seconds to milliseconds.

    Args:
        boundaries (list of float): Times in seconds.

    Returns:
        list of float: Times in milliseconds.
    """
    return [float(boundary) * 1000.0 for boundary in boundaries]