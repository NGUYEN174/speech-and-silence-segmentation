"""
Audio loading utilities.

Provides a single function to load WAV files with:
- Automatic stereo-to-mono conversion
- Integer PCM to float64 conversion
- Original sampling rate preservation (no resampling)
"""

import numpy as np
import soundfile as sf


def load_audio(path):
    """Load a WAV file and return the signal and sampling rate.
    
    Processing steps:
        1. Read WAV file using soundfile (supports PCM16, PCM24, PCM32, float32, float64)
        2. If multi-channel, convert to mono by averaging across channels
        3. Ensure the signal is float64 for consistent numerical processing
        4. Preserve the original sampling rate (no resampling)
    
    Integer PCM conversion:
        soundfile automatically converts integer PCM to floating-point
        in the range [-1.0, 1.0]. This is the standard convention.
        For example, a 16-bit PCM value of 32767 becomes approximately 1.0.
    
    Args:
        path (str): Path to the WAV file.
    
    Returns:
        tuple: (signal, fs)
            - signal (np.ndarray): 1D float64 array of audio samples
            - fs (int): Sampling rate in Hz
    
    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """
    # Read the WAV file
    # soundfile returns float64 by default for integer PCM formats
    signal, fs = sf.read(path, dtype='float64')
    
    # Convert multi-channel to mono by averaging
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
        # Note: axis=1 averages across channels (columns)
        # Each row is one time sample, each column is one channel
    
    # Ensure float64
    signal = signal.astype(np.float64)
    
    return signal, fs
