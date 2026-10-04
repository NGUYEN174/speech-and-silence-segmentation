"""
LAB file parser for ground-truth speech/silence annotations.

LAB files contain time-aligned annotations of speech and silence regions.

Supported format:
    start_time  end_time  label

Example:
    0.000  0.500  sil
    0.500  1.800  v
    1.800  2.200  uv
    2.200  3.000  sil

Times are in seconds.

For this project, the task is SPEECH/SILENCE segmentation.

Important:
    'v'  = voiced speech
    'uv' = unvoiced speech

Both 'v' and 'uv' belong to SPEECH.

Only labels such as 'sil' represent SILENCE.

The parser normalizes labels to:
    'speech'
    'silence'
"""

import os


# ============================================================
# LABEL MAPPING
# ============================================================
#
# The dataset uses:
#   v  = voiced speech
#   uv = unvoiced speech
#   sil = silence
#
# Therefore BOTH 'v' and 'uv' must be mapped to 'speech'
# for the Speech/Silence Segmentation task.
#
LABEL_MAP = {
    # --------------------------------------------------------
    # Speech variants
    # --------------------------------------------------------
    'speech': 'speech',
    'sp': 'speech',
    'v': 'speech',
    'voiced': 'speech',
    'uv': 'speech',
    'unvoiced': 'speech',
    '1': 'speech',

    # --------------------------------------------------------
    # Silence variants
    # --------------------------------------------------------
    'silence': 'silence',
    'sil': 'silence',
    '0': 'silence',
    'noise': 'silence',
    'n': 'silence',
}


# ============================================================
# READ LAB FILE
# ============================================================

def read_lab(path, label_map=None):
    """Read a LAB annotation file.

    Parses time-aligned annotations and normalizes labels
    into either 'speech' or 'silence'.

    Args:
        path (str):
            Path to the .lab file.

        label_map (dict, optional):
            Custom mapping from raw labels to normalized labels.
            If None, LABEL_MAP is used.

    Returns:
        list of dict:
            Each interval has:
                'start'  : float, start time in seconds
                'end'    : float, end time in seconds
                'label'  : 'speech' or 'silence'

    Raises:
        FileNotFoundError:
            If the LAB file does not exist.

        ValueError:
            If an unknown label is encountered.
    """

    # --------------------------------------------------------
    # Check file existence
    # --------------------------------------------------------

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ERROR: LAB file not found:\n  {path}"
        )

    # --------------------------------------------------------
    # Use default label mapping if no custom mapping is given
    # --------------------------------------------------------

    if label_map is None:
        label_map = LABEL_MAP

    intervals = []

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    with open(path, 'r', encoding='utf-8') as f:

        for line_num, line in enumerate(f, 1):

            line = line.strip()

            # ------------------------------------------------
            # Skip empty lines and comments
            # ------------------------------------------------

            if not line or line.startswith('#'):
                continue

            parts = line.split()

            # ------------------------------------------------
            # Ignore metadata lines
            #
            # Example:
            #   F0mean 145
            #   F0std 33.7
            #
            # These are not time-aligned annotations.
            # ------------------------------------------------

            if parts[0].lower() in ('f0mean', 'f0std'):
                continue

            # ------------------------------------------------
            # LAB annotation requires:
            #
            # start end label
            # ------------------------------------------------

            if len(parts) < 3:
                print(
                    f"  Warning: Skipping line {line_num} in {path}: "
                    f"expected 3+ columns, got {len(parts)}"
                )
                continue

            # ------------------------------------------------
            # Parse start time, end time and label
            # ------------------------------------------------

            try:
                start = float(parts[0])
                end = float(parts[1])
                raw_label = parts[2].lower().strip()

            except ValueError:
                print(
                    f"  Warning: Could not parse line {line_num} "
                    f"in {path}: {line}"
                )
                continue

            # ------------------------------------------------
            # Validate time interval
            # ------------------------------------------------

            if end < start:
                raise ValueError(
                    f"ERROR: Invalid interval on line {line_num} "
                    f"in {path}: end time ({end}) is smaller "
                    f"than start time ({start})."
                )

            # ------------------------------------------------
            # Normalize label
            # ------------------------------------------------

            normalized = label_map.get(raw_label)

            if normalized is None:
                raise ValueError(
                    f"ERROR: Unknown label '{raw_label}' "
                    f"on line {line_num} in {path}.\n"
                    f"Known labels: {list(label_map.keys())}\n"
                    f"Please update LABEL_MAP in "
                    f"src/common/lab_reader.py"
                )

            # ------------------------------------------------
            # Store interval
            # ------------------------------------------------

            intervals.append({
                'start': start,
                'end': end,
                'label': normalized,
            })

    return intervals


# ============================================================
# GET SPEECH BOUNDARIES
# ============================================================

def get_speech_boundaries(intervals):
    """Extract speech segment boundaries from LAB intervals.

    This function is specifically designed for the
    Speech/Silence Segmentation task.

    Important:
        Voiced ('v') and unvoiced ('uv') parts are both speech.

        Therefore:

            v   1.02 - 1.88
            uv  1.88 - 1.95
            v   1.95 - 2.16

        represents ONE continuous speech region:

            speech 1.02 - 2.16

    Adjacent or overlapping speech intervals are merged.

    Example:

        silence  0.00 - 1.02
        speech   1.02 - 1.88
        speech   1.88 - 1.95
        speech   1.95 - 2.16
        silence  2.16 - 2.50

        Returns:

            [1.02, 2.16]

    Args:
        intervals (list of dict):
            Output from read_lab().

    Returns:
        list of float:
            Flat list of speech boundaries:

            [start1, end1, start2, end2, ...]
    """

    speech_segments = []

    # --------------------------------------------------------
    # Find all speech intervals
    # --------------------------------------------------------

    for interval in intervals:

        if interval['label'] != 'speech':
            continue

        start = interval['start']
        end = interval['end']

        # ----------------------------------------------------
        # First speech interval
        # ----------------------------------------------------

        if not speech_segments:
            speech_segments.append([start, end])
            continue

        # ----------------------------------------------------
        # Last detected speech interval
        # ----------------------------------------------------

        last_start, last_end = speech_segments[-1]

        # ----------------------------------------------------
        # Merge adjacent or overlapping speech intervals
        #
        # Example:
        #
        # 1.02 - 1.88
        # 1.88 - 1.95
        #
        # becomes:
        #
        # 1.02 - 1.95
        #
        # This prevents v -> uv -> v transitions from being
        # incorrectly treated as speech/silence boundaries.
        # ----------------------------------------------------

        if start <= last_end:
            speech_segments[-1][1] = max(last_end, end)

        else:
            # ------------------------------------------------
            # A real silence gap exists between the two
            # speech intervals, so create a new segment.
            # ------------------------------------------------

            speech_segments.append([start, end])

    # --------------------------------------------------------
    # Convert segments to flat boundary list
    # --------------------------------------------------------

    boundaries = []

    for start, end in speech_segments:
        boundaries.append(start)
        boundaries.append(end)

    return boundaries


# ============================================================
# LABEL FRAMES FROM LAB
# ============================================================

def label_frames_from_lab(intervals, frame_center_times):
    """Assign speech/silence labels to each frame.

    Labeling rule:
        A frame is labeled according to its CENTER TIME.

        If the center time falls inside a speech interval:
            1 = speech

        Otherwise:
            0 = silence

    Important:
        Because 'v' and 'uv' are both normalized to 'speech',
        both voiced and unvoiced speech frames receive label 1.

    This function is used during training to generate
    ground-truth frame labels.

    It is NOT used during testing.

    Args:
        intervals (list of dict):
            Output of read_lab().

        frame_center_times (np.ndarray):
            Center time of each frame in seconds.

    Returns:
        np.ndarray:
            Binary labels:

                1 = speech
                0 = silence
    """

    import numpy as np

    # --------------------------------------------------------
    # Initially mark every frame as silence
    # --------------------------------------------------------

    labels = np.zeros(
        len(frame_center_times),
        dtype=int
    )

    # --------------------------------------------------------
    # Get all speech intervals
    #
    # Both v and uv are already normalized to 'speech'
    # by read_lab().
    # --------------------------------------------------------

    speech_intervals = [
        (iv['start'], iv['end'])
        for iv in intervals
        if iv['label'] == 'speech'
    ]

    # --------------------------------------------------------
    # Label each frame according to its CENTER TIME
    # --------------------------------------------------------

    for i, t in enumerate(frame_center_times):

        for start, end in speech_intervals:

            if start <= t < end:
                labels[i] = 1
                break

    return labels