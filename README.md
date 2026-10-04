# Speech/Silence Segmentation of Recorded Speech Signals

> University Assignment — Audio and Speech Processing

## 1. Project Overview

This project implements **three speech/silence segmentation algorithms** for a university audio processing course. Each algorithm processes WAV files to detect speech boundaries and separate speech from silence.

| Algorithm | Responsible | Status |
|---|---|---|
| **Binary Search** (Energy-based) | Student 1 | 🔲 Placeholder |
| **Histogram** (Giannakopoulos, 2014) | **This student** | ✅ Implemented |
| **Simple Statistics** | Student 3 | 🔲 Placeholder |

### One-Command Demonstration

```bash
python run_all.py
```

This single command runs **all 3 algorithms** on **all 4 test WAV files**, producing:
- **12 result figures** (3 algorithms × 4 files)
- **summary.csv** with MAE/RMSE for every combination
- **Final comparison table** showing which algorithm performs best

---

## 2. Assignment Requirements

- 3 algorithms × 4 test files = **12 result figures**
- Training data → determine parameters → freeze → test data → evaluation
- **Frame length = 25 ms** (fixed, NOT tunable)
- **Frame shift = 10 ms** (fixed, NOT tunable)
- **Minimum silence = 200 ms** (fixed, NOT tunable)
- Evaluation metrics: **MAE** and **RMSE** (in milliseconds)

---

## 3. Algorithms

### Algorithm 1: Binary Search (Energy-based)

**Reference:** CS425 Audio and Speech Processing, Matthieu Hodgkinson, 2012, Section 2.1

**Status:** Placeholder — waiting for the responsible student's implementation.

### Algorithm 2: Histogram (My Algorithm)

**Reference:** Theodoros Giannakopoulos, "A method for silence removal and segmentation of speech signals" (2014)

Uses **two features**: signal energy (log STE) and spectral centroid.

### Algorithm 3: Simple Statistics

**Reference:** Simple Statistics method from course/reference material.

**Status:** Placeholder — waiting for the responsible student's implementation.

---

## 4. Histogram Algorithm — Theory

### 4.1 Feature Extraction

**Short-Time Energy (STE):**

$$\text{STE}_i = \frac{1}{N} \sum_{n=0}^{N-1} x_i[n]^2$$

**Log STE:** (used as the energy feature)

$$\text{logSTE}_i = \log_{10}(\text{STE}_i + \epsilon)$$

where $\epsilon = 10^{-12}$ prevents $\log(0)$.

**Spectral Centroid:**

$$C_i = \frac{\sum_{k=0}^{N/2} f_k \cdot |X_i(k)|}{\sum_{k=0}^{N/2} |X_i(k)|}$$

where $f_k = k \cdot f_s / N$ is the frequency of bin $k$ and $|X_i(k)|$ is the magnitude spectrum.

### 4.2 Histogram Thresholding

For each feature sequence:

1. **Build histogram** of feature values across all frames
2. **Smooth** the histogram with a Gaussian filter (sigma = trained parameter)
3. **Find local maxima** (peaks) in the smoothed histogram
4. **Identify M1 and M2:**
   - **M1** = peak at the lower feature value (silence distribution)
   - **M2** = peak at the higher feature value (speech distribution)
5. **Compute threshold:**

$$T = \frac{W \cdot M_1 + M_2}{W + 1}$$

where $W > 1$ places the threshold **closer to the silence peak** (M1), making the classifier more conservative.

### 4.3 Frame Classification

A frame is classified as **SPEECH** if **both** conditions hold:

- $\text{logSTE}_i > T_{\text{energy}}$
- $C_i > T_{\text{centroid}}$

> **Note on uncertainty:** The exact AND vs OR combination rule is not fully specified in all versions of the reference. This implementation uses AND logic. The rule is isolated in `_combine_feature_decisions()` and can be easily changed.

### 4.4 Post-Processing

If a silence gap between two speech segments is shorter than **200 ms**, the segments are **merged** (the silence frames are reclassified as speech).

---

## 5. Fixed Parameters

| Parameter | Value | Source |
|---|---|---|
| Frame length | 25 ms | Lecturer (fixed) |
| Frame shift | 10 ms | Lecturer (fixed) |
| Min silence | 200 ms | Lecturer (fixed) |

> **These values MUST NOT be optimized.** They are assignment constraints.

Frame length/shift in samples are computed from each file's actual sampling rate:

```python
frame_length_samples = round(0.025 * fs)
frame_shift_samples  = round(0.010 * fs)
```

---

## 6. Training / Test Separation

> [!CAUTION]
> **Test labels must NEVER be used for parameter tuning.** This is data leakage.

**Correct workflow:**

```
Training WAV + Training LAB
    → extract features
    → tune parameters (bins, sigma, W, prominence, distance)
    → save trained_parameters.json

Test WAV (only)
    → predict using frozen parameters

Test LAB
    → evaluate ONLY (MAE/RMSE)
```

---

## 7. Directory Structure

```
Speech-Silence-Segmentation/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── .gitignore                         ← Excludes data and results
│
├── run_all.py                         ← Main demonstration script
├── train_all.py                       ← Training pipeline
│
├── config/
│   ├── __init__.py
│   ├── parameters.py                  ← Fixed & configurable parameters
│   └── trained_parameters.json        ← Generated by train_all.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── common/
│   │   ├── __init__.py
│   │   ├── audio.py                   ← WAV loading (mono, float64)
│   │   ├── framing.py                 ← Signal framing
│   │   ├── features.py                ← STE, log STE, spectral centroid
│   │   ├── lab_reader.py              ← LAB file parser
│   │   ├── segments.py                ← Mask ↔ segments ↔ boundaries
│   │   ├── postprocess.py             ← 200ms silence merging
│   │   ├── evaluation.py              ← MAE, RMSE, boundary matching
│   │   ├── plotting.py                ← Result figure generation
│   │   └── interfaces.py              ← Common algorithm interface
│   │
│   ├── binary_search/                 ← Algorithm 1 (placeholder)
│   │   ├── __init__.py
│   │   ├── algorithm.py
│   │   └── train.py
│   │
│   ├── histogram/                     ← Algorithm 2 (IMPLEMENTED)
│   │   ├── __init__.py
│   │   ├── algorithm.py               ← Histogram segmentation
│   │   └── train.py                   ← Parameter grid search
│   │
│   └── simple_statistics/             ← Algorithm 3 (placeholder)
│       ├── __init__.py
│       ├── algorithm.py
│       └── train.py
│
├── data/
│   ├── TinHieuHuanLuyen/              ← Training data (WAV + LAB)
│   └── TinHieuKiemThu/                ← Test data (WAV + LAB)
│
└── results/
    ├── binary_search/                 ← Algorithm 1 figures
    ├── histogram/                     ← Algorithm 2 figures
    ├── simple_statistics/             ← Algorithm 3 figures
    ├── snr/                           ← SNR experiment results
    └── summary.csv                    ← Combined results table
```

---

## 8. Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy ≥ 1.21
- scipy ≥ 1.7
- matplotlib ≥ 3.4
- soundfile ≥ 0.10
- pandas ≥ 1.3

---

## 9. Dataset Placement

Place the lecturer's audio files in the following directories:

**Training data:**
```
data/TinHieuHuanLuyen/
    ├── file1.wav
    ├── file1.lab
    ├── file2.wav
    ├── file2.lab
    └── ...
```

**Test data:**
```
data/TinHieuKiemThu/
    ├── test1.wav
    ├── test1.lab
    ├── test2.wav
    ├── test2.lab
    ├── test3.wav
    ├── test3.lab
    ├── test4.wav
    └── test4.lab
```

> [!IMPORTANT]
> The WAV/LAB files are provided by the lecturer and must NOT be committed to Git.

---

## 10. LAB File Format

Expected format (3 columns, whitespace-separated):

```
start_time  end_time  label
```

Example:
```
0.000  0.500  sil
0.500  1.800  speech
1.800  2.200  sil
2.200  3.400  speech
```

**Supported labels:**
| Raw Label | Normalized |
|---|---|
| `speech`, `sp`, `v`, `voiced`, `1` | speech |
| `silence`, `sil`, `unvoiced`, `uv`, `0`, `noise`, `n` | silence |

If your LAB files use different labels, update `LABEL_MAP` in `src/common/lab_reader.py`.

---

## 11. Commands

### Training

```bash
python train_all.py
```

Trains all algorithms using data from `data/TinHieuHuanLuyen/` and saves parameters to `config/trained_parameters.json`.

### Final Demonstration (ONE COMMAND)

```bash
python run_all.py
```

Runs all 3 algorithms on all 4 test files. Generates 12 figures and summary.csv.

### Histogram-Only Development Mode

```bash
python run_all.py --algorithm histogram
```

For debugging during development — runs only the Histogram algorithm.

### SNR Robustness Experiment

```bash
python run_all.py --snr
```

Adds noise at various SNR levels (20, 10, 5, 0 dB) and measures performance degradation.

### Standalone Histogram Training

```bash
python -m src.histogram.train
```

---

## 12. Result Figures

Each figure includes:
- **Waveform** with ground-truth (RED) and detected (BLUE) boundaries
- **Feature sequences** with thresholds
- **Algorithm-specific diagnostics**

### Histogram Figures (4 subplots each)

1. **Waveform** — RED = ground truth, BLUE = detected
2. **Feature curves** — log STE + spectral centroid with thresholds
3. **Energy histogram** — raw histogram, smoothed, M1, M2, threshold
4. **Centroid histogram** — raw histogram, smoothed, M1, M2, threshold

---

## 13. Evaluation Metrics

### MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |b_i^{detected} - b_i^{truth}|$$

### RMSE (Root Mean Squared Error)

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (b_i^{detected} - b_i^{truth})^2}$$

Both are reported in **milliseconds**.

---

## 14. SNR Robustness Experiment

Tests algorithm performance under varying noise conditions.

| Setting | Value |
|---|---|
| Noise type | White Gaussian |
| SNR levels | 20, 10, 5, 0 dB |
| Random seed | 42 (reproducible) |

**Purpose:** Evaluate whether segmentation degrades as SNR decreases.

Output: `results/snr/snr_results.csv` and `results/snr/snr_comparison.png`

> [!NOTE]
> The SNR experiment does NOT modify original lecturer files. Noisy signals are generated in memory.

---

## 15. Histogram Parameter Selection

### Trainable Parameters

| Parameter | Candidates | Description |
|---|---|---|
| `histogram_bins` | 30, 50, 70 | Number of histogram bins |
| `smoothing_sigma` | 2.0, 3.0, 5.0 | Gaussian smoothing sigma |
| `W` | 3.0, 5.0, 7.0, 10.0 | Threshold weighting |
| `peak_prominence` | 0.005, 0.01, 0.02 | Min peak prominence (fraction) |
| `peak_distance` | 3, 5, 8 | Min distance between peaks |

### Training Process

1. Pre-compute features for all training files
2. Grid search over all parameter combinations
3. For each combination: compute boundaries, compare to ground truth
4. Select combination with lowest **training MAE**
5. Save to `config/trained_parameters.json`

> [!WARNING]
> W is NOT automatically optimal. The value selected is whatever minimizes MAE on training data. Do not claim it is universally optimal.

---

## 16. How to Add Another Algorithm

### Step 1: Create a New Module

```
src/my_algorithm/
├── __init__.py
├── algorithm.py
└── train.py
```

### Step 2: Implement the Interface

```python
# src/my_algorithm/algorithm.py
from src.common.interfaces import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    
    @property
    def name(self):
        return "My Algorithm"
    
    @property
    def key(self):
        return "my_algorithm"
    
    def train(self, training_files, training_labs):
        # Extract features from training WAV files
        # Use training LAB files for ground-truth labels
        # Return trained parameters as a dict
        return {"threshold": 0.5}
    
    def predict(self, signal, fs, params):
        # Segment the signal using trained parameters
        return {
            'algorithm_name': self.name,
            'speech_mask': ...,
            'segments': ...,
            'boundaries_ms': ...,
            'feature_times': ...,
            'features': {...},
            'thresholds': {...},
            'diagnostics': {...},
        }
```

### Step 3: Register in interfaces.py

Edit `src/common/interfaces.py`, function `get_all_algorithms()`:

```python
from src.my_algorithm.algorithm import MyAlgorithm

return [
    BinarySearchAlgorithm(),
    HistogramAlgorithm(),
    SimpleStatisticsAlgorithm(),
    # MyAlgorithm(),  # ← Add here
]
```

**No other files need to be modified.**

---

## 17. Histogram Algorithm Traceability

| Reference Concept | Implementation |
|---|---|
| Signal framing | `src/common/framing.py` → `frame_signal()` |
| Signal energy (STE) | `src/common/features.py` → `calculate_ste()` |
| Log energy | `src/common/features.py` → `calculate_log_ste()` |
| Spectral centroid | `src/common/features.py` → `calculate_spectral_centroid()` |
| Histogram construction | `src/histogram/algorithm.py` → `compute_histogram_threshold()` |
| Histogram smoothing | `src/histogram/algorithm.py` → `gaussian_filter1d()` |
| Local maxima detection | `src/histogram/algorithm.py` → `find_peaks()` |
| M1 / M2 identification | `src/histogram/algorithm.py` → peak sorting logic |
| Threshold formula | `src/histogram/algorithm.py` → `T = (W*M1 + M2)/(W+1)` |
| Speech/silence decision | `src/histogram/algorithm.py` → `_combine_feature_decisions()` |
| Short silence merging | `src/common/postprocess.py` → `merge_short_silences()` |
| Parameter training | `src/histogram/train.py` → `train_histogram()` |

---

## 18. Presentation Structure (Histogram Student)

| Slide | Topic |
|---|---|
| 1 | Problem Definition |
| 2 | WAV, Sampling Rate, and Framing |
| 3 | Short-Time Energy / Signal Energy |
| 4 | Spectral Centroid |
| 5 | Histogram Construction |
| 6 | Histogram Smoothing + Local Maxima |
| 7 | M1, M2, and Threshold Formula |
| 8 | Speech/Silence Decision |
| 9 | 200 ms Short Silence Post-processing |
| 10 | Training and Parameter Selection |
| 11 | Test Results — 4 WAV Files |
| 12 | MAE / RMSE |
| 13 | Noise / SNR Experiment |
| 14 | Comparison with Other Two Algorithms |

---

## 19. Troubleshooting

### "No training WAV files found"
Place the lecturer's training files in `data/TinHieuHuanLuyen/`.

### "trained_parameters.json not found"
Run `python train_all.py` first.

### "Missing LAB file for test01.wav"
Ensure every `.wav` file has a matching `.lab` file with the same name.

### "Unknown label 'xyz'"
Update `LABEL_MAP` in `src/common/lab_reader.py` to include the new label.

### "Only 1 peak found for energy"
The histogram smoothing may be too strong or too weak. Check `smoothing_sigma` and `histogram_bins` values. Consider adjusting the parameter grid.

### Import errors
Run from the **project root directory** (the directory containing `run_all.py`).

---

## 20. Academic Integrity

- This project does **NOT** fabricate results.
- Placeholder algorithms return `None` and are **skipped** during evaluation.
- All parameters come from **training data only**.
- The Histogram implementation follows the Giannakopoulos (2014) reference.
- No WebRTC VAD, Silero VAD, neural networks, or unrelated methods are used.
- All design decisions and uncertainties are **documented**.
