"""
Histogram-based Speech/Silence Segmentation Algorithm.

Reference:
    Theodoros Giannakopoulos,
    "A method for silence removal and segmentation of speech signals"
    (2014)

This implementation follows the Giannakopoulos method:
    1. Frame the signal
    2. Compute two feature sequences:
       a. Signal energy (log Short-Time Energy)
       b. Spectral centroid
    3. For each feature:
       a. Build a histogram of feature values
       b. Smooth the histogram (Gaussian smoothing)
       c. Detect local maxima
       d. Identify M1 (first/lower peak) and M2 (second/higher peak)
       e. Compute threshold: T = (W * M1 + M2) / (W + 1)
    4. Classify each frame:
       A frame is speech if BOTH:
       - energy > energy_threshold  AND
       - spectral_centroid > centroid_threshold
    5. Apply 200ms minimum-silence post-processing
    6. Return speech segments and boundaries

IMPORTANT DESIGN DECISIONS AND UNCERTAINTIES:
    1. Feature combination rule:
       The implementation uses AND logic (both conditions must be true)
       for combining energy and spectral centroid decisions.
       This is isolated in the function _combine_feature_decisions()
       so it can be easily modified if the original source specifies
       a different rule.
    
    2. M1/M2 ordering:
       M1 is identified as the peak corresponding to silence/low values.
       M2 is identified as the peak corresponding to speech/high values.
       For energy: M1 < M2 (silence has lower energy)
       For spectral centroid: M1 < M2 (silence has lower centroid)
       The threshold is placed between them, weighted toward M1.
    
    3. Histogram bins and smoothing:
       These are tunable parameters determined during training.
       They are NOT assignment-fixed parameters.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from src.common.interfaces import BaseAlgorithm
from src.common.audio import load_audio
from src.common.framing import frame_signal
from src.common.features import calculate_ste, calculate_log_ste, calculate_spectral_centroid
from src.common.segments import mask_to_segments, segments_to_boundaries, boundaries_to_ms
from src.common.postprocess import merge_short_silences


class HistogramAlgorithm(BaseAlgorithm):
    """Histogram-based speech/silence segmentation.
    
    Implements the Giannakopoulos (2014) method using:
        - Signal energy (log STE) histogram
        - Spectral centroid histogram
        - Dual-threshold classification
    """
    
    @property
    def name(self):
        return "Histogram"
    
    @property
    def key(self):
        return "histogram"
    
    def train(self, training_files, training_labs):
        """Train Histogram parameters using training data.
        
        Delegates to the train module.
        
        Args:
            training_files (list of str): Paths to training WAV files.
            training_labs (list of str): Paths to training LAB files.
        
        Returns:
            dict: Trained histogram parameters.
        """
        from src.histogram.train import train_histogram
        return train_histogram(training_files, training_labs)
    
    def predict(self, signal, fs, params):
        """Segment a signal using the Histogram method.
        
        Args:
            signal (np.ndarray): 1D audio signal.
            fs (int): Sampling rate.
            params (dict): Trained histogram parameters.
        
        Returns:
            dict: Standardized result dictionary with all diagnostics.
        """
        # Extract parameters
        histogram_bins = params.get('histogram_bins', 50)
        smoothing_sigma = params.get('smoothing_sigma', 3.0)
        W = params.get('W', 5.0)
        peak_prominence = params.get('peak_prominence', 0.01)
        peak_distance = params.get('peak_distance', 5)
        
        # Step 1: Frame the signal
        frame_result = frame_signal(signal, fs)
        frames = frame_result['frames']
        feature_times = frame_result['center_times']
        
        if frame_result['num_frames'] == 0:
            return self._empty_result(feature_times)
        
        # Step 2: Compute feature sequences
        ste = calculate_ste(frames)
        log_ste = calculate_log_ste(ste)
        spectral_centroid = calculate_spectral_centroid(frames, fs)

        # Step 3a: Histogram thresholding for energy (log STE)
        energy_diag = compute_histogram_threshold(
            log_ste, histogram_bins, smoothing_sigma, W,
            peak_prominence, peak_distance,
            feature_name='energy (log STE)'
        )
        
        # Step 3b: Histogram thresholding for spectral centroid
        centroid_diag = compute_histogram_threshold(
            spectral_centroid, histogram_bins, smoothing_sigma, W,
            peak_prominence, peak_distance,
            feature_name='spectral centroid'
        )
        
        # Step 4: Classify each frame
        energy_threshold = energy_diag['threshold']
        centroid_threshold = centroid_diag['threshold']
        
        speech_mask = _combine_feature_decisions(
            log_ste, energy_threshold,
            spectral_centroid, centroid_threshold
        )
        
        # Step 5: Post-processing - merge short silences
        speech_mask = merge_short_silences(speech_mask, feature_times)
        
        # Step 6: Convert to segments and boundaries
        segments = mask_to_segments(speech_mask, feature_times)
        boundaries_s = segments_to_boundaries(segments)
        boundaries_ms = boundaries_to_ms(boundaries_s)
        
        # Build result dictionary
        return {
            'algorithm_name': self.name,
            'speech_mask': speech_mask,
            'segments': segments,
            'boundaries_ms': boundaries_ms,
            'feature_times': feature_times,
            'features': {
                'energy_log_ste': log_ste,
                'spectral_centroid': spectral_centroid,
            },
            'thresholds': {
                'energy': energy_threshold,
                'centroid': centroid_threshold,
            },
            'diagnostics': {
                # Feature sequences
                'energy_feature': log_ste,
                'spectral_centroid': spectral_centroid,
                
                # Energy histogram diagnostics
                'energy_histogram': energy_diag['histogram'],
                'energy_histogram_smoothed': energy_diag['histogram_smoothed'],
                'energy_histogram_bins': energy_diag['bin_edges'],
                'energy_peaks': energy_diag['peaks'],
                'energy_M1': energy_diag['M1'],
                'energy_M2': energy_diag['M2'],
                'energy_threshold': energy_threshold,
                
                # Spectral centroid histogram diagnostics
                'centroid_histogram': centroid_diag['histogram'],
                'centroid_histogram_smoothed': centroid_diag['histogram_smoothed'],
                'centroid_histogram_bins': centroid_diag['bin_edges'],
                'centroid_peaks': centroid_diag['peaks'],
                'centroid_M1': centroid_diag['M1'],
                'centroid_M2': centroid_diag['M2'],
                'centroid_threshold': centroid_threshold,
                
                # Parameters used
                'histogram_bins': histogram_bins,
                'smoothing_sigma': smoothing_sigma,
                'W': W,
                'peak_prominence': peak_prominence,
                'peak_distance': peak_distance,
            }
        }
    
    def _empty_result(self, feature_times):
        """Return an empty result when no frames are available."""
        return {
            'algorithm_name': self.name,
            'speech_mask': np.array([]),
            'segments': [],
            'boundaries_ms': [],
            'feature_times': feature_times,
            'features': {},
            'thresholds': {},
            'diagnostics': {},
        }


def compute_histogram_threshold(feature_values, num_bins, smoothing_sigma,
                                 W, peak_prominence, peak_distance,
                                 feature_name='feature'):
    """Compute a threshold from the histogram of a feature sequence.
    
    Implements the Giannakopoulos histogram thresholding method:
        1. Build a histogram of feature values
        2. Smooth the histogram with Gaussian filter
        3. Find local maxima (peaks)
        4. Identify M1 (lower/silence peak) and M2 (higher/speech peak)
        5. Compute threshold: T = (W * M1 + M2) / (W + 1)
    
    The threshold is weighted toward M1 (silence peak) because W > 1
    places the threshold closer to the silence distribution.
    
    Args:
        feature_values (np.ndarray): 1D array of feature values.
        num_bins (int): Number of histogram bins.
        smoothing_sigma (float): Gaussian smoothing sigma for histogram.
        W (float): Weighting parameter for threshold formula.
        peak_prominence (float): Minimum prominence for peak detection.
        peak_distance (int): Minimum distance between peaks (in bins).
        feature_name (str): Name of the feature (for error messages).
    
    Returns:
        dict with keys:
            'histogram' (np.ndarray): Raw histogram counts
            'histogram_smoothed' (np.ndarray): Smoothed histogram
            'bin_edges' (np.ndarray): Histogram bin edges
            'peaks' (np.ndarray): Indices of detected peaks
            'M1' (float or None): Lower peak location (feature value)
            'M2' (float or None): Higher peak location (feature value)
            'threshold' (float or None): Computed threshold
            'fallback_used' (bool): Whether a fallback was used
    """
    # Step 1: Build histogram
    histogram, bin_edges = np.histogram(feature_values, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Step 2: Smooth histogram
    histogram_float = histogram.astype(float)
    histogram_smoothed = gaussian_filter1d(histogram_float, sigma=smoothing_sigma)
    
    # Step 3: Find local maxima
    peaks, properties = find_peaks(
        histogram_smoothed,
        prominence=peak_prominence * np.max(histogram_smoothed),
        distance=peak_distance
    )
    
    # Step 4: Identify M1 and M2
    M1 = None
    M2 = None
    threshold = None
    fallback_used = False
    
    if len(peaks) >= 2:
        # Sort peaks by their bin center position (feature value)
        peak_positions = bin_centers[peaks]
        sorted_indices = np.argsort(peak_positions)
        
        # M1 = lowest peak (silence), M2 = second peak (speech)
        # Among all peaks, take the two most prominent ones
        # Then order them by feature value
        peak_heights = histogram_smoothed[peaks]
        
        # Select the two most prominent peaks
        if len(peaks) > 2:
            prominence_order = np.argsort(peak_heights)[::-1]  # descending
            top_two_indices = prominence_order[:2]
            top_two_positions = peak_positions[top_two_indices]
            sorted_top_two = np.sort(top_two_positions)
            M1 = float(sorted_top_two[0])
            M2 = float(sorted_top_two[1])
        else:
            sorted_positions = np.sort(peak_positions)
            M1 = float(sorted_positions[0])
            M2 = float(sorted_positions[1])
        
        # Step 5: Compute threshold
        # T = (W * M1 + M2) / (W + 1)
        threshold = (W * M1 + M2) / (W + 1)
    
    elif len(peaks) == 1:
        # FALLBACK: Only one peak found
        # Use median as a simple fallback
        print(f"  WARNING: Only 1 peak found for {feature_name}.")
        print(f"  Using fallback: threshold = median of feature values.")
        print(f"  This fallback is NOT the original Giannakopoulos method.")
        
        M1 = float(bin_centers[peaks[0]])
        threshold = float(np.median(feature_values))
        fallback_used = True
    
    else:
        # FALLBACK: No peaks found
        print(f"  ERROR: No peaks found in histogram for {feature_name}.")
        print(f"  Using fallback: threshold = median of feature values.")
        print(f"  This fallback is NOT the original Giannakopoulos method.")
        
        threshold = float(np.median(feature_values))
        fallback_used = True
    
    return {
        'histogram': histogram,
        'histogram_smoothed': histogram_smoothed,
        'bin_edges': bin_edges,
        'peaks': peaks,
        'M1': M1,
        'M2': M2,
        'threshold': threshold,
        'fallback_used': fallback_used,
    }


def _combine_feature_decisions(energy_values, energy_threshold,
                                centroid_values, centroid_threshold):
    """Combine energy and spectral centroid decisions.
    
    Decision rule:
        A frame is classified as SPEECH if BOTH:
            energy_value > energy_threshold
            AND
            centroid_value > centroid_threshold
    
    IMPORTANT NOTE ON UNCERTAINTY:
        The exact combination rule (AND vs OR) is not fully specified
        in all versions of the Giannakopoulos reference.
        
        This implementation uses AND logic because:
        - It is more conservative (fewer false positives)
        - It requires evidence from both features
        
        If the lecturer's course material specifies a different rule,
        modify ONLY this function. No other code needs to change.
    
    Args:
        energy_values (np.ndarray): Energy feature values per frame.
        energy_threshold (float): Energy threshold.
        centroid_values (np.ndarray): Spectral centroid values per frame.
        centroid_threshold (float): Spectral centroid threshold.
    
    Returns:
        np.ndarray: Binary mask (1 = speech, 0 = silence).
    """
    if energy_threshold is None or centroid_threshold is None:
        # If either threshold is missing, use only the available one
        if energy_threshold is not None:
            return (energy_values > energy_threshold).astype(int)
        elif centroid_threshold is not None:
            return (centroid_values > centroid_threshold).astype(int)
        else:
            return np.zeros(len(energy_values), dtype=int)
    
    energy_speech = energy_values > energy_threshold
    centroid_speech = centroid_values > centroid_threshold
    
    # AND logic: both conditions must be true
    speech_mask = (energy_speech & centroid_speech).astype(int)
    
    return speech_mask
