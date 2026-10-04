"""
Huan luyen Algorithm 1 (Binary Search).

Voi moi cau hinh dac trung (kind, use_log, smooth_len): gop dac trung cac khung
noi (S) va lang (Q) theo nhan chuan cua TAP HUAN LUYEN, tim nguong T bang tim
kiem nhi phan. Sau do duyet luoi cac sieu tham so hau xu ly (hyst, expand,
min_speech_ms) va chon bo co sai so bien nho nhat tren tap huan luyen.

Chi dung nhan cua tap huan luyen. Tham so khung 25/10 ms va min silence 200 ms
la co dinh (xem algorithm.py), KHONG nam trong luoi.

Chay rieng:  python -m src.binary_search.train
"""
import glob
import itertools
import json
import os
import wave

import numpy as np

from src.binary_search.algorithm import (
    preprocess, compute_feature_track, frame_params, num_frames, frame_centers,
    train_threshold, predict_file)

# Luoi sieu tham so (nho de chay nhanh; co the chinh)
GRID = {
    "kind": ["MA", "RMS"],
    "use_log": [False, True],
    "smooth_len": [1, 5, 9],
    "hyst": [0.0, 0.15],
    "expand": [0, 2, 4],
    "min_speech_ms": [0, 50, 100],
}

DEFAULT_TRAIN_DIR = os.path.join("data", "TinHieuHuanLuyen")

# Theo README de bai: sil = lang; v (huu thanh) va uv (vo thanh) deu la TIENG NOI.
SIL_LABELS = {"sil"}


# ---------------------------------------------------------------------
# Doc du lieu
# ---------------------------------------------------------------------
def read_wav(path):
    """Doc WAV PCM 8/16/24/32 bit -> (x float64 mono trong [-1, 1], fs)."""
    with wave.open(path, "rb") as w:
        n_ch, sw = w.getnchannels(), w.getsampwidth()
        fs, n = w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 2 ** 23, v - 2 ** 24, v)
        x = v.astype(np.float64) / 2 ** 23
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2 ** 31
    else:
        raise ValueError("Khong ho tro do rong mau %d byte" % sw)
    if n_ch > 1:
        x = x[: len(x) // n_ch * n_ch].reshape(-1, n_ch).mean(axis=1)
    return x, fs


def read_lab(path):
    """Doc .lab (t_dau t_cuoi nhan) -> [(t_dau, t_cuoi, is_speech)]; bo dong F0mean/F0std."""
    segs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            tok = line.strip().split()
            if len(tok) < 3:
                continue
            try:
                s, e = float(tok[0]), float(tok[1])
            except ValueError:
                continue
            segs.append((s, e, tok[2].strip().lower() not in SIL_LABELS))
    return segs


def lab_boundaries(segs):
    """Bien chuan = diem doi nhan noi/lang (v <-> uv khong tinh)."""
    return [segs[i][0] for i in range(1, len(segs)) if segs[i][2] != segs[i - 1][2]]


def lab_frame_mask(segs, centers):
    """Nhan chuan theo khung (True = tieng noi) tai tam khung."""
    mask = np.zeros(len(centers), dtype=bool)
    for s, e, is_sp in segs:
        if is_sp:
            mask[(centers >= s) & (centers < e)] = True
    return mask


def boundary_errors(gt_b, pr_b, cap_ms):
    """Khoang cach (ms) tu moi bien chuan toi bien du doan gan nhat va nguoc lai."""
    gt, pr = np.asarray(gt_b, float), np.asarray(pr_b, float)
    if len(gt) == 0:
        return np.array([]), np.full(len(pr), cap_ms)
    if len(pr) == 0:
        return np.full(len(gt), cap_ms), np.array([])
    d = np.abs(gt[:, None] - pr[None, :]) * 1000.0
    return d.min(axis=1), d.min(axis=0)


def load_one(wav_path, lab_path):
    """Nap mot cap wav + lab va tinh san cac dai luong dung khi huan luyen."""
    x, fs = read_wav(wav_path)
    dur = len(x) / fs
    segs = read_lab(lab_path)
    frame_len, hop = frame_params(fs)
    nf = num_frames(len(x), frame_len, hop)
    centers = frame_centers(nf, frame_len, hop, fs)
    return {"name": os.path.splitext(os.path.basename(wav_path))[0],
            "x": x, "xn": preprocess(x), "fs": fs, "dur": dur,
            "gt_b": lab_boundaries(segs), "gt_mask": lab_frame_mask(segs, centers)}


def load_dataset(training_files, training_labs):
    """Nap danh sach cap (wav, lab). Hai danh sach phai cung do dai, cung thu tu."""
    if len(training_files) != len(training_labs):
        raise ValueError("So file WAV (%d) khac so file LAB (%d)"
                         % (len(training_files), len(training_labs)))
    return [load_one(w, l) for w, l in zip(training_files, training_labs)]


def find_pairs(folder):
    """Tim cac cap *.wav + *.lab cung ten trong thu muc (file thieu .lab bi bo qua)."""
    if not os.path.isdir(folder):
        raise SystemExit("LOI: khong tim thay thu muc du lieu: %s" % os.path.abspath(folder))
    wavs = sorted(set(glob.glob(os.path.join(folder, "*.wav"))
                      + glob.glob(os.path.join(folder, "*.WAV"))))
    files, labs = [], []
    for w in wavs:
        lab = os.path.splitext(w)[0] + ".lab"
        if os.path.exists(lab):
            files.append(w)
            labs.append(lab)
        else:
            print("  [bo qua] thieu file .lab cho", os.path.basename(w))
    if not files:
        raise SystemExit("LOI: khong co cap .wav + .lab nao trong %s" % folder)
    return files, labs


# ---------------------------------------------------------------------
# Huan luyen
# ---------------------------------------------------------------------
def pooled_class_features(data, tracks):
    """Gop dac trung khung cua moi file thanh S (noi) va Q (lang) theo nhan chuan."""
    S = np.concatenate([t[d["gt_mask"][: len(t)]] for d, t in zip(data, tracks)])
    Q = np.concatenate([t[~d["gt_mask"][: len(t)]] for d, t in zip(data, tracks)])
    return S, Q


def objective(data, params):
    """Diem so (ms, cang nho cang tot) = trung binh MAE(chuan->du doan) va MAE(du doan->chuan).
    Thanh phan thu hai phat viec sinh ra bien thua."""
    e_gt, e_pr = [], []
    for d in data:
        res = predict_file(d["x"], d["fs"], params)
        a, b = boundary_errors(d["gt_b"], res["boundaries"], d["dur"] * 1000.0)
        e_gt.append(a)
        e_pr.append(b)
    e_gt, e_pr = np.concatenate(e_gt), np.concatenate(e_pr)
    m1 = np.mean(e_gt) if len(e_gt) else 0.0
    m2 = np.mean(e_pr) if len(e_pr) else 0.0
    return 0.5 * (m1 + m2)


def grid_search(data, verbose=True):
    """Tim bo tham so tot nhat; tra ve (best_params, best_score_ms)."""
    best, best_score = None, float("inf")

    for kind, use_log, sm in itertools.product(GRID["kind"], GRID["use_log"],
                                               GRID["smooth_len"]):
        tracks = [compute_feature_track(d["xn"], d["fs"], kind, use_log, sm) for d in data]
        S, Q = pooled_class_features(data, tracks)
        T, sep = train_threshold(S, Q)          # tim kiem nhi phan tren dac trung gop

        for hy, ex, ms in itertools.product(GRID["hyst"], GRID["expand"],
                                            GRID["min_speech_ms"]):
            params = {"kind": kind, "use_log": use_log, "smooth_len": sm,
                      "T": float(T), "sep": float(sep), "hyst": hy,
                      "expand": ex, "min_speech_ms": ms}
            score = objective(data, params)
            if score < best_score:
                best, best_score = params, score

        if verbose:
            print("  %-3s log=%-5s smooth=%d  T=%.5f | tot nhat toi nay: %.1f ms"
                  % (kind, use_log, sm, T, best_score))
    return best, best_score


def train_binary_search(training_files, training_labs, verbose=True):
    """Huan luyen tu danh sach duong dan WAV va LAB cua tap huan luyen.

    Tra ve (params, train_score_ms); params la dict JSON-able:
    kind, use_log, smooth_len, T, sep, hyst, expand, min_speech_ms.
    """
    data = load_dataset(training_files, training_labs)
    if not data:
        raise ValueError("Tap huan luyen rong")
    if verbose:
        print("Binary Search: huan luyen tren %d file" % len(data))
    params, score = grid_search(data, verbose)
    if verbose:
        print("Bo tham so toi uu:", params, "| diem huan luyen: %.2f ms" % score)
    return params, score


if __name__ == "__main__":
    files, labs = find_pairs(DEFAULT_TRAIN_DIR)
    p, s = train_binary_search(files, labs)
    print(json.dumps(p, indent=2))