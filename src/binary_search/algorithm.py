"""
Algorithm 1: Binary Search (Energy-based) - CS425, Hodgkinson 2012, Section 2.1

Y tuong: tim nguong T tren dac trung nang luong (MA / RMS) sao cho hai loai
nham lan (tieng noi bi coi la lang, lang bi coi la tieng noi) can bang nhau,
F(T) = 0, bang tim kiem nhi phan. Sau do gan nhan khung, hau xu ly, doi ra bien.

File nay chua phan THUAT TOAN (tien xu ly, dac trung, tim nguong, phan doan,
lop BinarySearchAlgorithm). Phan HUAN LUYEN nam trong train.py.
Chi dung numpy.
"""
import numpy as np

from src.common.interfaces import BaseAlgorithm

# ---------------- Tham so co dinh theo de bai (KHONG duoc toi uu) ----------------
FRAME_MS = 25.0     # do dai khung (ms)
HOP_MS = 10.0       # do dich khung (ms)
MIN_SIL_MS = 200.0  # do dai toi thieu cua 1 khoang lang (ms)
EPS_LOG = 1e-6      # hang so nho tranh log(0)


# =====================================================================
# 1. Tien xu ly + dac trung
# =====================================================================
def preprocess(x):
    """Bo DC roi chuan hoa bien do dinh ve 1 (tin hieu toan 0 thi giu nguyen)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    return x / peak if peak > 0 else x


def frame_params(fs):
    """Doi 25 ms / 10 ms sang so mau theo tan so lay mau cua tung file."""
    return int(round(FRAME_MS * fs / 1000.0)), int(round(HOP_MS * fs / 1000.0))


def num_frames(n_samples, frame_len, hop):
    """So khung (khung cuoi nam tron trong tin hieu; toi thieu 1 khung)."""
    if n_samples < frame_len:
        return 1
    return 1 + (n_samples - frame_len) // hop


def frame_centers(n_frames, frame_len, hop, fs):
    """Thoi diem tam cua tung khung (giay)."""
    return (np.arange(n_frames) * hop + frame_len / 2.0) / fs


def frame_features(x, fs, kind="MA"):
    """Dac trung theo khung bang tong tich luy (khong tao ma tran khung).

    kind: 'MA' = mean(|x|), 'RMS' = sqrt(mean(x^2)), 'STE' = sum(x^2)
    """
    frame_len, hop = frame_params(fs)
    if len(x) < frame_len:
        x = np.concatenate((x, np.zeros(frame_len - len(x))))
    nf = num_frames(len(x), frame_len, hop)
    starts = np.arange(nf) * hop
    ends = starts + frame_len

    if kind == "MA":
        c = np.concatenate(([0.0], np.cumsum(np.abs(x))))
        return (c[ends] - c[starts]) / frame_len

    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    ste = np.maximum(c2[ends] - c2[starts], 0.0)
    if kind == "STE":
        return ste
    if kind == "RMS":
        return np.sqrt(ste / frame_len)
    raise ValueError("kind phai la MA, RMS hoac STE")


def moving_average(a, length):
    """Lam tron bang trung binh truot (cua so le, lap gia tri bien)."""
    if length <= 1:
        return a
    h = length // 2
    p = np.concatenate((np.full(h, a[0]), a, np.full(h, a[-1])))
    c = np.concatenate(([0.0], np.cumsum(p)))
    return (c[length:] - c[:-length]) / length


def compute_feature_track(x_norm, fs, kind, use_log, smooth_len):
    """Dac trung theo khung -> lam tron -> (tuy chon) log10."""
    feat = moving_average(frame_features(x_norm, fs, kind), smooth_len)
    if use_log:
        feat = np.log10(feat + EPS_LOG)
    return feat


# =====================================================================
# 2. Tim nguong bang tim kiem nhi phan
# =====================================================================
def find_overlap_region(S, Q, trim=1.0):
    """Vung chong lan [lo, hi]: lo = phan vi trim% cua S, hi = phan vi (100-trim)% cua Q."""
    return np.percentile(S, trim), np.percentile(Q, 100.0 - trim)


def confusion_error(T, f, g):
    """F(T) = (ti le khung lang > T) - (ti le khung noi <= T); F giam don dieu theo T."""
    n_f_below = int(np.sum(f <= T))
    n_g_above = int(np.sum(g > T))
    return n_g_above / len(g) - n_f_below / len(f), (n_f_below, n_g_above)


def binary_search_threshold(f, g, lo, hi, eps_rel=1e-4, max_iter=60, patience=3):
    """Tim T sao cho F(T) = 0 bang tim kiem nhi phan tren [lo, hi]."""
    t_min, t_max = lo, hi
    eps = eps_rel * (hi - lo)
    T = 0.5 * (t_min + t_max)
    last, same = None, 0

    for _ in range(max_iter):
        F, counts = confusion_error(T, f, g)
        if F > 0:       # nhieu khung lang nam tren T -> T qua thap
            t_min = T
        else:
            t_max = T
        if t_max - t_min < eps:
            break
        same = same + 1 if counts == last else 0
        last = counts
        if same >= patience:
            break
        T = 0.5 * (t_min + t_max)
    return 0.5 * (t_min + t_max)


def train_threshold(S, Q, trim=1.0):
    """Nguong T tu dac trung khung noi (S) va khung lang (Q) cua tap huan luyen.

    Tra ve (T, sep) voi sep = mean(S) - mean(Q) (dung de dat do rong tre).
    """
    S, Q = np.asarray(S), np.asarray(Q)
    sep = float(np.mean(S) - np.mean(Q))
    lo, hi = find_overlap_region(S, Q, trim)
    if lo >= hi:                         # hai lop khong chong lan: lay diem giua khe ho
        return 0.5 * (lo + hi), sep
    f = S[(S >= lo) & (S <= hi)]
    g = Q[(Q >= lo) & (Q <= hi)]
    if len(f) == 0 or len(g) == 0:
        return 0.5 * (lo + hi), sep
    return binary_search_threshold(f, g, lo, hi), sep


# =====================================================================
# 3. Gan nhan khung + hau xu ly + doi ra bien
# =====================================================================
def classify_frames(feat, T, sep, hyst):
    """True = tieng noi. hyst > 0: nguong kep (bat = T + hyst*sep, tat = T - hyst*sep)."""
    if hyst <= 0:
        return feat > T
    t_on, t_off = T + hyst * sep, T - hyst * sep
    mask = np.zeros(len(feat), dtype=bool)
    state = False
    for k, v in enumerate(feat.tolist()):
        if state:
            if v < t_off:
                state = False
        elif v > t_on:
            state = True
        mask[k] = state
    return mask


def dilate(mask, e):
    """Mo rong vung tieng noi them e khung moi phia."""
    if e <= 0:
        return mask
    n = len(mask)
    c = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    lo = np.maximum(np.arange(n) - e, 0)
    hi = np.minimum(np.arange(n) + e + 1, n)
    return (c[hi] - c[lo]) > 0


def get_runs(mask):
    """Tach mask thanh cac doan (khung_dau, khung_cuoi_khong_gom, nhan)."""
    m = mask.astype(np.int8)
    change = np.nonzero(m[1:] != m[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(m)]))
    return [(int(s), int(e), bool(m[s])) for s, e in zip(starts, ends)]


def remove_short_runs(mask, value, min_frames, include_edges=True):
    """Dao nhan cac doan nhan `value` ngan hon min_frames."""
    out = mask.copy()
    runs = get_runs(mask)
    for i, (s, e, v) in enumerate(runs):
        if v != value or (e - s) >= min_frames:
            continue
        if (i == 0 or i == len(runs) - 1) and not include_edges:
            continue
        out[s:e] = not value
    return out


def mask_to_segments(mask, fs, n_samples):
    """Mask khung -> (boundaries [giay], segments [(t_dau, t_cuoi, 'speech'|'sil')])."""
    frame_len, hop = frame_params(fs)
    dur = n_samples / fs
    runs = get_runs(mask)
    boundaries = [(s * hop - hop / 2.0 + frame_len / 2.0) / fs for s, _, _ in runs[1:]]
    edges = [0.0] + boundaries + [dur]
    segments = [(edges[i], edges[i + 1], "speech" if runs[i][2] else "sil")
                for i in range(len(runs))]
    return boundaries, segments


def predict_file(x, fs, params):
    """Toan bo quy trinh phan doan cho mot tin hieu (xem BinarySearchAlgorithm.predict)."""
    xn = preprocess(x)
    feat = compute_feature_track(xn, fs, params["kind"], params["use_log"],
                                 params["smooth_len"])
    mask_raw = classify_frames(feat, params["T"], params["sep"], params["hyst"])

    # hau xu ly: mo rong bien, bo doan noi qua ngan, roi bo lang < 200 ms (de bai)
    mask_exp = dilate(mask_raw, int(params["expand"]))
    min_sp = int(np.ceil(params["min_speech_ms"] / HOP_MS))
    mask_exp = remove_short_runs(mask_exp, True, min_sp)
    min_sil = int(np.ceil(MIN_SIL_MS / HOP_MS))
    mask = remove_short_runs(mask_exp, False, min_sil)

    boundaries, segments = mask_to_segments(mask, fs, len(x))
    return {"xn": xn, "feat": feat, "mask_raw": mask_raw, "mask_exp": mask_exp,
            "mask": mask, "boundaries": boundaries, "segments": segments}


# =====================================================================
# 4. Lop thuat toan theo interface chung cua repo
# =====================================================================
class BinarySearchAlgorithm(BaseAlgorithm):
    """Algorithm 1: Binary Search (Energy-based)."""

    @property
    def name(self):
        return "Binary Search"

    @property
    def key(self):
        return "binary_search"

    def train(self, training_files, training_labs):
        """Huan luyen tren tap train (WAV + LAB), tra ve dict tham so (JSON-able)."""
        from src.binary_search.train import train_binary_search  # tranh import vong
        params, _ = train_binary_search(training_files, training_labs)
        return params

    def predict(self, signal, fs, params):
        """Phan doan `signal` bang tham so da dong bang (khong dung nhan test)."""
        res = predict_file(signal, fs, params)
        feat = res["feat"]
        frame_len, hop = frame_params(fs)
        times = frame_centers(len(feat), frame_len, hop, fs)
        return {
            "algorithm_name": self.name,
            "speech_mask": res["mask"],
            "segments": res["segments"],                       # (giay, giay, 'speech'/'sil')
            "boundaries_ms": [b * 1000.0 for b in res["boundaries"]],
            "feature_times": times,
            "features": {params["kind"] + (" (log10)" if params["use_log"] else ""): feat},
            "thresholds": {"T": params["T"],
                           "T_on": params["T"] + params["hyst"] * params["sep"],
                           "T_off": params["T"] - params["hyst"] * params["sep"]},
            "diagnostics": {"mask_raw": res["mask_raw"], "mask_exp": res["mask_exp"],
                            "xn": res["xn"], "params": dict(params)},
        }