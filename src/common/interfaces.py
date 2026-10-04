"""
Common algorithm interface for Speech/Silence Segmentation.

All algorithms MUST implement this interface so that:
    1. The common pipeline (run_all.py) can call them uniformly
    2. New algorithms can be added without modifying the pipeline
    3. Results are in a consistent format

To add a new algorithm:
    1. Create a new directory under src/ (e.g., src/my_algorithm/)
    2. Implement algorithm.py with a class inheriting BaseAlgorithm
    3. Implement train.py with a train() function
    4. Register the algorithm in get_all_algorithms()
"""

from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """Base class for all speech/silence segmentation algorithms.
    
    Every algorithm must implement:
        - name (str): Human-readable algorithm name
        - train(): Learn parameters from training data
        - predict(): Segment a test signal
    
    The predict() method must return a standardized result dictionary.
    """
    
    @property
    @abstractmethod
    def name(self):
        """Human-readable name of the algorithm.
        
        Returns:
            str: Algorithm name (e.g., 'Histogram', 'Binary Search')
        """
        pass
    
    @property
    @abstractmethod
    def key(self):
        """Machine-readable key for file paths and JSON.
        
        Returns:
            str: Algorithm key (e.g., 'histogram', 'binary_search')
        """
        pass
    
    @abstractmethod
    def train(self, training_files, training_labs):
        """Train the algorithm using training data.
        
        This method should:
            1. Extract features from training WAV files
            2. Use training LAB files for ground-truth labels
            3. Determine/optimize algorithm-specific parameters
            4. Return the trained parameters as a dictionary
        
        Args:
            training_files (list of str): Paths to training WAV files.
            training_labs (list of str): Paths to corresponding LAB files.
        
        Returns:
            dict: Trained parameters to be saved in trained_parameters.json.
                Must be JSON-serializable.
        """
        pass
    
    @abstractmethod
    def predict(self, signal, fs, params):
        """Apply the algorithm to segment a signal.
        
        Args:
            signal (np.ndarray): 1D audio signal (float64, mono).
            fs (int): Sampling rate in Hz.
            params (dict): Trained parameters for this algorithm
                (loaded from trained_parameters.json).
        
        Returns:
            dict: Standardized result dictionary with keys:
                'algorithm_name' (str): Name of the algorithm
                'speech_mask' (np.ndarray): Binary frame-level mask
                    (1 = speech, 0 = silence)
                'segments' (list of dict): Speech segments with
                    'start' and 'end' keys (in seconds)
                'boundaries_ms' (list of float): Flat boundary list
                    in milliseconds [start1, end1, start2, end2, ...]
                'feature_times' (np.ndarray): Time axis for features
                    (in seconds)
                'features' (dict): Feature arrays used by this algorithm
                'thresholds' (dict): Threshold values used
                'diagnostics' (dict): Algorithm-specific diagnostic data
                    for visualization
        """
        pass
    
    @property
    def is_placeholder(self):
        """Whether this is a placeholder (not yet implemented).
        
        Placeholder algorithms will be skipped during evaluation.
        
        Returns:
            bool: True if this is a placeholder.
        """
        return False


class PlaceholderAlgorithm(BaseAlgorithm):
    """Placeholder for algorithms not yet implemented.
    
    Used for Binary Search and Simple Statistics until the responsible
    students provide their implementations.
    
    This class:
        - Does NOT generate fake results
        - Does NOT generate fake MAE/RMSE
        - Does NOT pretend to be a real implementation
        - Clearly indicates that implementation is pending
    """
    
    def __init__(self, algorithm_name, algorithm_key):
        self._name = algorithm_name
        self._key = algorithm_key
    
    @property
    def name(self):
        return self._name
    
    @property
    def key(self):
        return self._key
    
    @property
    def is_placeholder(self):
        return True
    
    def train(self, training_files, training_labs):
        """Placeholder training - returns empty parameters."""
        print(f"\n  [{self.name}] Not yet implemented (placeholder).")
        print(f"  Waiting for the responsible student's implementation.")
        return {}
    
    def predict(self, signal, fs, params):
        """Placeholder prediction - returns None to indicate no results."""
        return None


def get_all_algorithms():
    """Get instances of all registered algorithms.
    
    This is the CENTRAL REGISTRATION POINT for algorithms.
    To add a new algorithm, import it here and add it to the list.
    
    When a student provides their implementation:
        1. Replace the PlaceholderAlgorithm with the real class
        2. No other files need to be modified
    
    Returns:
        list of BaseAlgorithm: All algorithm instances.
    """
    from src.binary_search.algorithm import BinarySearchAlgorithm
    from src.histogram.algorithm import HistogramAlgorithm
    from src.simple_statistics.algorithm import SimpleStatisticsAlgorithm
    
    return [
        BinarySearchAlgorithm(),
        HistogramAlgorithm(),
        SimpleStatisticsAlgorithm(),
    ]
