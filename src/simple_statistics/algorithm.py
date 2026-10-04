"""
Simple Statistics-based Speech/Silence Segmentation Algorithm.

Implementation of the Simple Statistics method using Gaussian distribution modeling of STE.
"""

import numpy as np

from src.common.interfaces import BaseAlgorithm
from src.common.framing import frame_signal
from src.common.features import calculate_ste
from src.common.segments import mask_to_segments, segments_to_boundaries, boundaries_to_ms
from src.common.postprocess import merge_short_silences


class SimpleStatisticsAlgorithm(BaseAlgorithm):
    """Simple Statistics speech/silence segmentation algorithm."""
    
    @property
    def name(self):
        return "Simple Statistics"
    
    @property
    def key(self):
        return "simple_statistics"
    
    def train(self, training_files, training_labs):
        """Train Simple Statistics parameters using training data."""
        from src.simple_statistics.train import train_simple_statistics
        return train_simple_statistics(training_files, training_labs)
    
    def predict(self, signal, fs, params):
        """Segment a signal using the Simple Statistics method."""
        threshold = params.get("threshold", 0.1)
        
        # Step 1: Frame the signal
        frame_result = frame_signal(signal, fs)
        frames = frame_result["frames"]
        feature_times = frame_result["center_times"]
        
        if frame_result["num_frames"] == 0:
            return {
                "algorithm_name": self.name,
                "speech_mask": np.array([]),
                "segments": [],
                "boundaries_ms": [],
                "feature_times": np.array([]),
                "features": {"ste": np.array([])},
                "thresholds": {"threshold": threshold},
                "diagnostics": {}
            }
            
        # Step 2: Compute normalized Short-Time Energy (STE)
        ste = calculate_ste(frames)
        max_ste = np.max(ste)
        if max_ste > 0:
            ste_norm = ste / max_ste
        else:
            ste_norm = ste
            
        # Step 3: Threshold classification
        speech_mask = (ste_norm > threshold).astype(int)
        
        # Step 4: Post-processing (merge short silences < 200 ms)
        speech_mask = merge_short_silences(speech_mask, feature_times)
        
        # Step 5: Convert to segments and boundaries
        segments = mask_to_segments(speech_mask, feature_times)
        boundaries_s = segments_to_boundaries(segments)
        boundaries_ms = boundaries_to_ms(boundaries_s)
        
        return {
            "algorithm_name": self.name,
            "speech_mask": speech_mask,
            "segments": segments,
            "boundaries_ms": boundaries_ms,
            "feature_times": feature_times,
            "features": {
                "ste": ste_norm,
            },
            "thresholds": {
                "threshold": threshold,
            },
            "diagnostics": {
                "ste": ste_norm,
                "threshold": threshold,
            }
        }
