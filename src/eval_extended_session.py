"""
Analyzes the full 10h live extended session (real speech through the real
INMP441, not loudspeaker playback) as ONE continuous stream -- no per-position
resets, matching genuine deployed behavior more closely than any prior test
tonight.

Ground truth: each button press marks a wake-word utterance. Since the
protocol is press-THEN-speak, the wake window is [press-2s, press+5s] (2s
margin before for reaction-time/clock variance, 5s after for the full
utterance). Any model fire inside ANY press's window counts as a detection;
everything else counts as a false alarm.

Same matched-detection-sensitivity methodology as tonight's earlier 7-position
analysis: per-arm threshold calibrated to a shared target detection count,
to avoid the FRR confound of comparing arms at a single shared threshold.
"""
import os, glob, json, wave
import numpy as np
import librosa
import tensorflow as tf
from scipy.stats import binomtest

SR = 16000; N_FFT = 512; HOP = 320; LOOKBACK = N_FFT - HOP
N_MELS = 32; FMIN = 60; FMAX = 6000; TW = 50
PCEN_ALPHA, PCEN_DELTA, PCEN_R, PCEN_TIME_C = 0.90, 2.0, 0.5, 0.10
ALPHA_EMA, TH_DOWN, MARGIN_MIN, COOLDOWN = 0.3, 0.30, 0.20, 7
DEFAULT_TH_UP = 0.60
STRIDE = int(0.3 * SR)

WAKE_BEFORE_S = 2.0
WAKE_AFTER_S = 5.0

SESSION_DIR = os.path.expanduser("~/wuwexp/domainshift/extended_session_10h")
MODELS_DIR = os.path.expanduser("~/wuwexp/models/ladder")
SESSION_START_EPOCH = 1784094114.0


def load_ch0_raw(path, channels=2, sampwidth=2):
    """Reads raw PCM directly, skipping the WAV header and trusting actual
    file size rather than the header's declared data-size field. Necessary
    because segment 1's PCM payload is exactly 2^31 bytes -- one byte over
    what a standard WAV header's 32-bit signed size field can hold, so the
    header's declared size silently overflowed and `wave.getnframes()`
    misreports it (confirmed: wave.open + readframes raised "buffer size
    must be a multiple of element size" on this exact file)."""
    file_size = os.path.getsize(path)
    HEADER_SIZE = 44  # standard canonical WAV header (fmt + data chunk headers, no extra chunks)
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        raw = f.read(file_size - HEADER_SIZE)
    frame_bytes = channels * sampwidth
    usable_len = (len(raw) // frame_bytes) * frame_bytes
    if usable_len != len(raw):
        print(f"  WARNING: {path} trailing {len(raw)-usable_len} bytes dropped (partial frame)")
    dt = {2: np.int16, 4: np.int32}[sampwidth]
    a = np.frombuffer(raw[:usable_len], dt).astype(np.float32).reshape(-1, channels)[:, 0]
    a = a / (2.0 ** (8 * sampwidth - 1))
    return a


def load_ch0_concat(paths):
    """Loads and concatenates raw mono channel-0 audio across multiple wav
    files, matching the same RMS-normalization convention as every other
    ladder-model evaluation tonight (load_ch0 in eval_pooled_significance.py)."""
    chunks = []
    for path in paths:
        a = load_ch0_raw(path)
        chunks.append(a)
        print(f"  loaded {path}: {len(a)/SR:.1f}s")
    full = np.concatenate(chunks)
    full = full / (np.sqrt((full ** 2).mean()) + 1e-9) * 0.05
    return full


def warm_zi(n=5):
    sil = np.zeros(1 * SR, dtype=np.float32)
    m = librosa.feature.melspectrogram(y=sil, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0, center=False)
    zi = None
    for _ in range(n):
        _, zi = librosa.pcen(S=m * (2 ** 31), sr=SR, hop_length=HOP, time_constant=PCEN_TIME_C,
                              gain=PCEN_ALPHA, bias=PCEN_DELTA, power=PCEN_R, eps=1e-3, zi=zi, return_zf=True)
    return zi


def stream_feats(audio):
    """Streams PCEN continuously across the WHOLE session -- one long-running
    adaptive state, same as genuine always-on deployment, not reset per
    position like tonight's earlier position-based tests."""
    zi = warm_zi()
    mel_buf = np.zeros((TW, N_MELS), dtype=np.float32)
    lookback = np.zeros(LOOKBACK, dtype=np.float32)
    out = []
    n_strides = (len(audio) - STRIDE) // STRIDE
    for k in range(n_strides):
        s = k * STRIDE
        chunk = audio[s:s + STRIDE].astype(np.float32)
        inp = np.concatenate([lookback, chunk])
        mel = librosa.feature.melspectrogram(y=inp, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                              n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0, center=False)
        lookback = chunk[-LOOKBACK:].astype(np.float32).copy() if len(chunk) >= LOOKBACK \
            else np.concatenate([lookback, chunk])[-LOOKBACK:].astype(np.float32)
        mel, zi = librosa.pcen(S=mel * (2 ** 31), sr=SR, hop_length=HOP, time_constant=PCEN_TIME_C,
                                gain=PCEN_ALPHA, bias=PCEN_DELTA, power=PCEN_R, eps=1e-3, zi=zi, return_zf=True)
        nf = mel.T
        n_new = nf.shape[0]
        if n_new >= TW:
            mel_buf = nf[-TW:].astype(np.float32)
        elif n_new > 0:
            mel_buf = np.roll(mel_buf, -n_new, axis=0)
            mel_buf[-n_new:] = nf
        out.append(mel_buf.copy())
        if (k + 1) % 20000 == 0:
            print(f"    feature progress: {k+1}/{n_strides} strides ({(k+1)*0.3/3600:.2f}h)", flush=True)
    return np.array(out, dtype=np.float32)


def run_decision_with_wake_mask(preds, th_up, wake_windows):
    """One continuous EMA+Schmitt+cooldown pass across the full session.
    Every fire is classified det (falls in a wake window) or FA (doesn't)."""
    ema = np.zeros(preds.shape[1], dtype=np.float32)
    cd, armed = 0, True
    det, fa = 0, 0
    fire_times = []
    for i in range(preds.shape[0]):
        p = preds[i]
        ema = ALPHA_EMA * p + (1 - ALPHA_EMA) * ema
        ew = float(ema[0]); margin = ew - float(np.max(ema[1:]))
        if not armed and ew < TH_DOWN:
            armed = True
        if cd > 0:
            cd -= 1
        if cd == 0 and armed and ew > th_up and margin > MARGIN_MIN:
            t = i * STRIDE / SR  # seconds into session
            is_wake = any(lo <= t <= hi for lo, hi in wake_windows)
            if is_wake:
                det += 1
            else:
                fa += 1
            fire_times.append((t, is_wake))
            armed = False; cd = COOLDOWN
    return det, fa, fire_times


def find_matched_threshold(preds, target_det, wake_windows, lo=0.50, hi=0.95, precision=0.002, max_iter=20):
    """Binary search on threshold -- det count is monotonically non-increasing
    as threshold rises, so this is valid and ~10x cheaper than a linear scan
    at the same precision."""
    best = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        det, fa, _ = run_decision_with_wake_mask(preds, mid, wake_windows)
        if best is None or abs(det - target_det) < abs(best[1] - target_det):
            best = (mid, det, fa)
        if det == target_det:
            return mid, det, fa
        elif det > target_det:
            lo = mid
        else:
            hi = mid
        if hi - lo < precision:
            break
    return best


def main():
    print("[1/4] Loading button presses...")
    presses = []
    with open(os.path.join(SESSION_DIR, "button_presses.log")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            presses.append(float(line))
    wake_windows = [(p - SESSION_START_EPOCH - WAKE_BEFORE_S, p - SESSION_START_EPOCH + WAKE_AFTER_S)
                    for p in presses]
    total_wake_s = sum(hi - lo for lo, hi in wake_windows)
    print(f"  {len(presses)} presses, {total_wake_s:.1f}s total wake-window time")

    print("\n[2/4] Loading + extracting streaming features (full session, continuous)...")
    wav_paths = sorted(glob.glob(os.path.join(SESSION_DIR, "live_extended*.wav")))
    print(f"  files: {wav_paths}")
    audio = load_ch0_concat(wav_paths)
    total_hours = len(audio) / SR / 3600
    print(f"  total audio: {total_hours:.2f}h")
    feats = stream_feats(audio)
    print(f"  {len(feats)} strides extracted")

    print("\n[3/4] Loading models + predicting...")
    models = {}
    for mp in sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras"))):
        arm = os.path.splitext(os.path.basename(mp))[0]
        models[arm] = tf.keras.models.load_model(mp, compile=False)
    print(f"  arms: {list(models)}")
    preds = {}
    for arm, m in models.items():
        preds[arm] = m.predict(feats, batch_size=1024, verbose=0)
        print(f"  {arm}: predicted")

    print("\n[4/4] Matched-threshold calibration + det/FA classification...")
    default_results = {arm: run_decision_with_wake_mask(preds[arm], DEFAULT_TH_UP, wake_windows) for arm in models}
    default_det = {arm: default_results[arm][0] for arm in models}
    target_det = min(default_det.values())
    print(f"  default (@0.60) det counts: {default_det} -> target = {target_det}")

    neg_hours = total_hours - total_wake_s / 3600.0
    results = {}
    for arm in models:
        th, det, fa = find_matched_threshold(preds[arm], target_det, wake_windows)
        fa_per_hr = fa / neg_hours if neg_hours > 0 else None
        results[arm] = {"th": round(float(th), 3), "det": det, "fa": fa, "fa_per_hr": round(fa_per_hr, 4)}
        print(f"  {arm:12s} th={th:.3f} det={det:4d} fa={fa:4d} fa/hr={fa_per_hr:.4f}")

    print(f"\n{'='*60}")
    print(f"Session: {total_hours:.2f}h total, {neg_hours:.2f}h negative-eligible, {len(presses)} wake presses")
    for arm in results:
        print(f"  {arm}: det={results[arm]['det']} fa={results[arm]['fa']} fa/hr={results[arm]['fa_per_hr']}")

    if "A0_conv" in results and "A2_twin" in results:
        k, n = results["A0_conv"]["fa"], results["A0_conv"]["fa"] + results["A2_twin"]["fa"]
        if n > 0:
            r = binomtest(k, n, p=0.5, alternative="two-sided")
            print(f"\n  conventional vs response-matched (k={k}, n={n}): p={r.pvalue:.4f}")
        else:
            print("\n  n=0, no test possible")

    out = {"total_hours": round(total_hours, 3), "neg_hours": round(neg_hours, 3),
           "n_presses": len(presses), "default_det": default_det, "target_det": target_det,
           "results": results}
    out_path = os.path.join(SESSION_DIR, "extended_session_results.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
