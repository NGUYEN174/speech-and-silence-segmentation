"""
LAB file parser for ground-truth speech/silence annotations.

LAB files contain time-aligned annotations of speech and silence regions.
The parser supports common LAB formats and provides configurable label mapping.

Supported format (auto-detected):
    start_time  end_time  label
    
    Example:
        0.000  0.500  sil
        0.500  1.800  speech
        1.800  2.200  sil

    Times are in seconds. Fields are separated by whitespace (spaces or tabs).

Label mapping:
    The parser normalizes labels to either 'speech' or 'silence'.
    Common label variants are handled:
        Speech:  'speech', 'sp', 'v', 'voiced', '1'
        Silence: 'silence', 'sil', 'unvoiced', 'uv', '0', 'noise', 'n'
    
    The mapping is configurable via the LABEL_MAP parameter.
"""

import os

# Default label mapping
# Maps various label strings to normalized 'speech' or 'silence'
LABEL_MAP = {
    # Speech variants
    'speech': 'speech',
    'sp': 'speech',
    'v': 'speech',
    'voiced': 'speech',
    '1': 'speech',
    
    # Silence variants
    'silence': 'silence',
    'sil': 'silence',
    'unvoiced': 'silence',
    'uv': 'silence',
    '0': 'silence',
    'noise': 'silence',
    'n': 'silence',
}


def read_lab(path, label_map=None):
    """Read a LAB annotation file.
    
    Parses time-aligned annotations and normalizes labels.
    
    Args:
        path (str): Path to the .lab file.
        label_map (dict, optional): Custom label mapping dictionary.
            Keys are raw label strings (lowercase), values are
            normalized labels ('speech' or 'silence').
            Defaults to the module-level LABEL_MAP.
    
    Returns:
        list of dict: Each dict has keys:
            'start' (float): Start time in seconds
            'end' (float): End time in seconds  
            'label' (str): Normalized label ('speech' or 'silence')
    
    Raises:
        FileNotFoundError: If the LAB file does not exist.
        ValueError: If a label is not found in the label map.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ERROR: LAB file not found:\n  {path}"
        )
    
    if label_map is None:
        label_map = LABEL_MAP
    
    intervals = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            if len(parts) < 3:
                # Try 2-column format: start label (end inferred from next)
                # For now, require at least 3 columns
                print(f"  Warning: Skipping line {line_num} in {path}: "
                      f"expected 3+ columns, got {len(parts)}")
                continue
            
            try:
                start = float(parts[0])
                end = float(parts[1])
                raw_label = parts[2].lower().strip()
            except ValueError:
                print(f"  Warning: Could not parse line {line_num} in {path}: "
                      f"{line}")
                continue
            
            # Normalize label
            normalized = label_map.get(raw_label)
            if normalized is None:
                raise ValueError(
                    f"ERROR: Unknown label '{raw_label}' on line {line_num} "
                    f"in {path}.\n"
                    f"Known labels: {list(label_map.keys())}\n"
                    f"Please update LABEL_MAP in src/common/lab_reader.py"
                )
            
            intervals.append({
                'start': start,
                'end': end,
                'label': normalized,
            })
    
    return intervals


def get_speech_boundaries(intervals):
    """Extract speech segment boundaries from LAB intervals.
    
    Converts interval annotations into a flat list of speech boundaries.
    Only boundaries of speech segments are included.
    
    Example:
        intervals:
            silence  0.0 - 0.5
            speech   0.5 - 1.8
            silence  1.8 - 2.2
            speech   2.2 - 3.4
        
        Returns: [0.5, 1.8, 2.2, 3.4]  (in seconds)
    
    Args:
        intervals (list of dict): Output of read_lab().
    
    Returns:
        list of float: Speech boundaries in seconds.
            [start1, end1, start2, end2, ...]
    """
    boundaries = []
    for interval in intervals:
        if interval['label'] == 'speech':
            boundaries.append(interval['start'])
            boundaries.append(interval['end'])
    return boundaries


def label_frames_from_lab(intervals, frame_center_times):
    """Assign speech/silence labels to each frame based on LAB annotations.
    
    Labeling rule:
        Each frame is labeled based on its CENTER TIME.
        If the center time of a frame falls within a speech interval,
        the frame is labeled as speech (1). Otherwise, silence (0).
    
    This is equivalent to "center-point" labeling, which is simpler
    and more robust than overlap-based labeling for this assignment.
    
    Note: This function is used ONLY during training to create
    ground-truth frame labels. It is NOT used during testing.
    
    Args:
        intervals (list of dict): Output of read_lab().
        frame_center_times (np.ndarray): Center time of each frame (seconds).
    
    Returns:
        np.ndarray: Binary array where 1 = speech, 0 = silence.
            Same length as frame_center_times.
    """
    import numpy as np
    
    labels = np.zeros(len(frame_center_times), dtype=int)
    
    # Get speech intervals
    speech_intervals = [
        (iv['start'], iv['end']) for iv in intervals if iv['label'] == 'speech'
    ]
    
    # Label each frame based on its center time
    for i, t in enumerate(frame_center_times):
        for start, end in speech_intervals:
            if start <= t < end:
                labels[i] = 1
                break
    
    return labels
