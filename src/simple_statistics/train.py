"""
Simple Statistics algorithm training module.

Calculates Short-Time Energy (STE) for framed audio signals, matches frame times
with groundtruth LAB annotations (speech vs. silence), calculates mean and std
of STE for speech and silence regions assuming Gaussian distributions, and finds
the optimal decision threshold T_optimal at the intersection of the two PDFs.
"""

import numpy as np

from src.common.audio import load_audio
from src.common.framing import frame_signal
from src.common.features import calculate_ste
from src.common.lab_reader import read_lab


def train_simple_statistics(training_files, training_labs):
    """Train Simple Statistics parameters using training data.
    
    Args:
        training_files (list of str): Paths to training WAV files.
        training_labs (list of str): Paths to training LAB files.
    
    Returns:
        dict: Trained parameters to be saved in trained_parameters.json.
            {
                "threshold": float,
                "mean_silence": float,
                "std_silence": float,
                "mean_speech": float,
                "std_speech": float,
            }
    """
    print("\n" + "=" * 50)
    print("SIMPLE STATISTICS TRAINING")
    print("=" * 50)
    print(f"Training files: {len(training_files)}")
    
    speech_stes = []
    silence_stes = []
    
    for wav_path, lab_path in zip(training_files, training_labs):
        try:
            signal, fs = load_audio(wav_path)
            lab_data = read_lab(lab_path)
        except Exception as e:
            print(f"  Warning: Could not read {wav_path} / {lab_path}: {e}")
            continue
            
        frame_result = frame_signal(signal, fs)
        frames = frame_result["frames"]
        feature_times = frame_result["center_times"]
        
        if frame_result["num_frames"] == 0:
            continue
            
        ste = calculate_ste(frames)
        max_ste = np.max(ste)
        if max_ste > 0:
            ste_norm = ste / max_ste
        else:
            ste_norm = ste
            
        for t, ste_val in zip(feature_times, ste_norm):
            is_speech = False
            for seg in lab_data["segments"]:
                if seg["start"] <= t <= seg["end"]:
                    if seg["label"] == "speech":
                        is_speech = True
                    break
            
            if is_speech:
                speech_stes.append(ste_val)
            else:
                silence_stes.append(ste_val)
                
    if len(speech_stes) == 0 or len(silence_stes) == 0:
        print("  Warning: Could not extract enough speech/silence frames. Using default threshold 0.1.")
        return {
            "threshold": 0.1,
            "mean_silence": 0.02,
            "std_silence": 0.01,
            "mean_speech": 0.35,
            "std_speech": 0.15,
        }
        
    mean_sp = float(np.mean(speech_stes))
    std_sp = float(np.std(speech_stes))
    mean_sil = float(np.mean(silence_stes))
    std_sil = float(np.std(silence_stes))
    
    if (std_sil + std_sp) > 0:
        threshold = (mean_sil * std_sp + mean_sp * std_sil) / (std_sil + std_sp)
    else:
        threshold = (mean_sil + mean_sp) / 2.0
        
    threshold = float(threshold)
    
    print(f"  Silence Distribution : Mean = {mean_sil:.4f}, Std = {std_sil:.4f}")
    print(f"  Speech Distribution  : Mean = {mean_sp:.4f}, Std = {std_sp:.4f}")
    print(f"  Optimal Threshold T  : {threshold:.4f}\n")
    
    return {
        "threshold": threshold,
        "mean_silence": mean_sil,
        "std_silence": std_sil,
        "mean_speech": mean_sp,
        "std_speech": std_sp,
    }
