"""
Room3 physical re-recording analysis, CORRECTED to use the same model family
(ladder/*.keras: A0_conv, A1_generic, A2_twin, A3_randtwin) and feature/
decision pipeline as eval_stream_arms.py, which is what actually produced the
published tab:twin numbers (verified against results/stream_P1_0.3m.json).
The first version of this script wrongly used the DSCNN TFLite arms from a
DIFFERENT experiment (the 413.7h speaker-disjoint test) and produced
degenerate all-zero detections as a result.

Adds matched-threshold calibration on top of the original pipeline: sweeps
TH_UP per arm so each arm's wake-stream detection count is brought down to
the same shared target (the min across arms at the default threshold),
before counting false alarms -- this is the fix for the FRR-confound issue
already flagged in the paper for the 413.7h comparison.
"""
import os, glob, json, wave
import numpy as np
import librosa
import tensorflow as tf

SR = 16000; N_FFT = 512; HOP = 320; LOOKBACK = N_FFT - HOP
N_MELS = 32; FMIN = 60; FMAX = 6000; TW = 50
PCEN_ALPHA, PCEN_DELTA, PCEN_R, PCEN_TIME_C = 0.90, 2.0, 0.5, 0.10
ALPHA_EMA, TH_DOWN, MARGIN_MIN, COOLDOWN = 0.3, 0.30, 0.20, 7
DEFAULT_TH_UP = 0.60
STRIDE = int(0.3 * SR)

MODELS_DIR = os.path.expanduser("~/wuwexp/models/ladder")
POSITIONS_DIR = os.path.expanduser("~/wuwexp/domainshift/phase3_room3")


def load_ch0(path):
    """Matches eval_stream_arms.py's load_ch0 exactly (verified against the
    published stream_P1_0.3m.json numbers)."""
    w = wave.open(path, "rb")
    sw, ch, n = w.getsampwidth(), w.getnchannels(), w.getnframes()
    dt = {2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(w.readframes(n), dt).astype(np.float32).reshape(-1, ch)[:, 0]
    a = a / (2.0 ** (8 * sw - 1))
    a = a / (np.sqrt((a ** 2).mean()) + 1e-9) * 0.05
    return a


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
    zi = warm_zi()
    mel_buf = np.zeros((TW, N_MELS), dtype=np.float32)
    lookback = np.zeros(LOOKBACK, dtype=np.float32)
    out = []
    for s in range(0, len(audio) - STRIDE + 1, STRIDE):
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
    return np.array(out, dtype=np.float32)


def count_acts(preds, th_up):
    ema = np.zeros(preds.shape[1], dtype=np.float32)
    cd, armed, acts = 0, True, 0
    for p in preds:
        ema = ALPHA_EMA * p + (1 - ALPHA_EMA) * ema
        ew = float(ema[0]); margin = ew - float(np.max(ema[1:]))
        if not armed and ew < TH_DOWN:
            armed = True
        if cd > 0:
            cd -= 1
        if cd == 0 and armed and ew > th_up and margin > MARGIN_MIN:
            acts += 1; armed = False; cd = COOLDOWN
    return acts


def find_matched_threshold(wake_preds, target_det, th_grid):
    best_th, best_det = th_grid[0], count_acts(wake_preds, th_grid[0])
    for th in th_grid:
        det = count_acts(wake_preds, th)
        if det <= target_det:
            return th, det
        best_th, best_det = th, det
    return best_th, best_det


def main():
    positions = sorted(glob.glob(os.path.join(POSITIONS_DIR, "room3_*")))
    print(f"Positions: {[os.path.basename(p) for p in positions]}")

    print("\n[1/3] Extracting streaming features...")
    feats = {}
    for pos_dir in positions:
        pos = os.path.basename(pos_dir)
        feats[pos] = {}
        for cls in ["wake", "speech", "noise"]:
            a = load_ch0(os.path.join(pos_dir, f"{cls}.wav"))
            feats[pos][cls] = stream_feats(a)
            print(f"  {pos}/{cls}: {len(feats[pos][cls])} strides")

    print("\n[2/3] Loading ladder models + computing predictions...")
    arm_preds = {}
    model_paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))
    print(f"  models found: {[os.path.basename(p) for p in model_paths]}")
    for mp in model_paths:
        arm = os.path.splitext(os.path.basename(mp))[0]
        m = tf.keras.models.load_model(mp, compile=False)
        arm_preds[arm] = {}
        for pos_dir in positions:
            pos = os.path.basename(pos_dir)
            arm_preds[arm][pos] = {
                cls: m.predict(feats[pos][cls], batch_size=512, verbose=0)
                for cls in ["wake", "speech", "noise"]
            }
        print(f"  {arm}: done")

    print("\n[3/3] Sanity check against known-good phase3 P1_0.3m result...")
    sanity_dir = os.path.expanduser("~/wuwexp/domainshift/phase3/P1_0.3m")
    if os.path.isdir(sanity_dir):
        a = load_ch0(os.path.join(sanity_dir, "wake.wav"))
        f = stream_feats(a)
        for mp in model_paths:
            arm = os.path.splitext(os.path.basename(mp))[0]
            m = tf.keras.models.load_model(mp, compile=False)
            det = count_acts(m.predict(f, batch_size=512, verbose=0), DEFAULT_TH_UP)
            print(f"  sanity {arm:16s} det@0.60 = {det}  (published: see stream_P1_0.3m.json)")

    print("\n[4/4] Matched-threshold calibration + FA counting on room3...")
    th_grid = np.arange(0.50, 0.96, 0.01)
    results = {}
    for pos_dir in positions:
        pos = os.path.basename(pos_dir)
        neg_hours = (len(feats[pos]["speech"]) + len(feats[pos]["noise"])) * (STRIDE / SR) / 3600.0
        default_det = {arm: count_acts(arm_preds[arm][pos]["wake"], DEFAULT_TH_UP) for arm in arm_preds}
        target_det = min(default_det.values())
        print(f"\n  {pos}: default @0.60 = {default_det} -> target = {target_det}")

        results[pos] = {"neg_hours": round(neg_hours, 3), "target_det": target_det, "arms": {}}
        for arm in arm_preds:
            th, det = find_matched_threshold(arm_preds[arm][pos]["wake"], target_det, th_grid)
            sfa = count_acts(arm_preds[arm][pos]["speech"], th)
            nfa = count_acts(arm_preds[arm][pos]["noise"], th)
            fa_per_hr = (sfa + nfa) / neg_hours if neg_hours > 0 else None
            results[pos]["arms"][arm] = {
                "matched_th_up": round(float(th), 3), "det": det,
                "speech_fa": sfa, "noise_fa": nfa, "fa_per_hr": round(fa_per_hr, 2),
            }
            print(f"    {arm:16s} th={th:.2f} det={det:3d}  speechFA={sfa:3d} noiseFA={nfa:3d}  FA/hr={fa_per_hr:.2f}")

    out_path = os.path.join(os.path.dirname(POSITIONS_DIR), "room3_matched_results.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
