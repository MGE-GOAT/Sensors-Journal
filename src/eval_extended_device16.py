"""
CORRECTED 10h live extended-session analysis using a FIXED multiplicative
gain (not adaptive per-stream RMS-forcing) to bring raw INMP441 capture into
the model's expected operating range.

Why: the device firmware's real MIC_SCALE is a fixed constant (1/2^27 for
S32 capture, empirically tuned so real speech lands at rms~=0.086-0.087,
"matches the training-data range (librosa-loaded int16 audio, rms~=0.05-0.10)"
-- see main.cpp comments). eval_stream_arms.py approximates this with a
per-clip adaptive RMS-normalize-to-0.05 step, which works fine for short,
fairly uniform streams (confirmed: reproduces published stream_P1_0.3m.json
numbers) but is NOT equivalent to a fixed gain for a long, highly
heterogeneous stream like this 10h session (mostly silence + rare loud
speech) -- forcing the GLOBAL average to 0.05 over-amplifies the sparse loud
content far more than a true fixed gain would.

Fix: apply the EXACT device scaling. The deployed device (USE_S32_CAPTURE=1)
computes float = S32_sample * (1/2^27). We recorded with `arecord -f S16_LE`,
and ALSA derives S16 from the mic's S32 by taking the top 16 bits
(S16 = S32 / 2^16). Our Python loads that S16 as S16/2^15 = S32/2^31. To
match the device's S32/2^27, the exact multiplier is 2^31 / 2^27 = 2^4 = 16.
This is the bit-exact reconstruction of what the deployed model actually
sees -- not an empirical fudge. Independent sanity check: P1_0.3m raw
rms 0.0045 * 16 = 0.072, right at the device's empirically-calibrated
operating level (~0.087 for close conversational speech).
"""
import os, glob, json
import numpy as np
import tensorflow as tf
from scipy.stats import binomtest

import sys
sys.path.insert(0, os.path.expanduser("~/wuwexp"))
import eval_canonical as C

FIXED_GAIN = 16.0  # EXACT device logic: S32*(1/2^27) reconstructed from our S16 capture = *2^4

SESSION_DIR = os.path.expanduser("~/wuwexp/domainshift/extended_session_10h")
MODELS_DIR = os.path.expanduser("~/wuwexp/models/ladder")
SESSION_START_EPOCH = 1784094114.0
WAKE_BEFORE_S, WAKE_AFTER_S = 2.0, 5.0
DEFAULT_TH_UP = 0.60
STRIDE = C.STRIDE
SR = C.SR

OUT_PATH = os.path.join(SESSION_DIR, "extended_session_results_DEVICE_GAIN16.json")


def load_ch0_concat_fixed_gain(paths):
    chunks = []
    for path in paths:
        a = C.load_ch0_raw(path, channels=2, sampwidth=2)
        chunks.append(a)
        print(f"  loaded {path}: {len(a)/SR:.1f}s raw_rms={np.sqrt(np.mean(a**2)):.6f}")
    full = np.concatenate(chunks)
    full = full * FIXED_GAIN  # FIXED multiplicative gain, NOT adaptive RMS-forcing
    print(f"  post-gain rms: {np.sqrt(np.mean(full**2)):.6f} (fixed gain={FIXED_GAIN}x)")
    return full


def run_decision_with_wake_mask(preds, th_up, wake_windows):
    ema = np.zeros(preds.shape[1], dtype=np.float32)
    cd, armed = 0, True
    det, fa = 0, 0
    for i in range(preds.shape[0]):
        p = preds[i]
        ema = C.ALPHA_EMA * p + (1 - C.ALPHA_EMA) * ema
        ew = float(ema[0]); margin = ew - float(np.max(ema[1:]))
        if not armed and ew < C.TH_DOWN:
            armed = True
        if cd > 0:
            cd -= 1
        if cd == 0 and armed and ew > th_up and margin > C.MARGIN_MIN:
            t = i * STRIDE / SR
            is_wake = any(lo <= t <= hi for lo, hi in wake_windows)
            if is_wake:
                det += 1
            else:
                fa += 1
            armed = False; cd = C.COOLDOWN
    return det, fa


def find_matched_threshold(preds, target_det, wake_windows, lo=0.50, hi=0.95, precision=0.002, max_iter=20):
    best = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        det, fa = run_decision_with_wake_mask(preds, mid, wake_windows)
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

    print("\n[2/4] Loading + extracting streaming features (FIXED GAIN, continuous)...")
    wav_paths = sorted(glob.glob(os.path.join(SESSION_DIR, "live_extended*.wav")))
    audio = load_ch0_concat_fixed_gain(wav_paths)
    total_hours = len(audio) / SR / 3600
    print(f"  total audio: {total_hours:.2f}h")
    feats = C.stream_feats(audio)
    print(f"  {len(feats)} strides extracted")

    print("\n[3/4] Loading models + predicting...")
    models = {}
    for mp in sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras"))):
        arm = os.path.splitext(os.path.basename(mp))[0]
        models[arm] = tf.keras.models.load_model(mp, compile=False)
    print(f"  arms: {list(models)}")
    preds = {arm: m.predict(feats, batch_size=1024, verbose=0) for arm, m in models.items()}
    print("  predicted all arms")

    print("\n[4/4] Matched-threshold calibration + det/FA classification...")
    default_det = {arm: run_decision_with_wake_mask(preds[arm], DEFAULT_TH_UP, wake_windows)[0] for arm in models}
    default_fa = {arm: run_decision_with_wake_mask(preds[arm], DEFAULT_TH_UP, wake_windows)[1] for arm in models}
    target_det = min(default_det.values())
    print(f"  default (@0.60) det counts: {default_det} -> target = {target_det}")
    print(f"  default (@0.60) fa counts: {default_fa}")

    neg_hours = total_hours - total_wake_s / 3600.0
    results = {}
    if target_det > 0:
        for arm in models:
            th, det, fa = find_matched_threshold(preds[arm], target_det, wake_windows)
            fa_per_hr = fa / neg_hours if neg_hours > 0 else None
            results[arm] = {"th": round(float(th), 3), "det": det, "fa": fa, "fa_per_hr": round(fa_per_hr, 4)}
            print(f"  {arm:12s} th={th:.3f} det={det:4d} fa={fa:4d} fa/hr={fa_per_hr:.4f}")
    else:
        print("  target_det=0 -- matched-threshold calibration would be degenerate; "
              "reporting DEFAULT-threshold (unmatched) numbers only, honestly disclosed as such.")
        for arm in models:
            fa_per_hr = default_fa[arm] / neg_hours if neg_hours > 0 else None
            results[arm] = {"th": DEFAULT_TH_UP, "det": default_det[arm], "fa": default_fa[arm],
                             "fa_per_hr": round(fa_per_hr, 4), "unmatched": True}

    print(f"\n{'='*60}")
    print(f"Session: {total_hours:.2f}h total, {neg_hours:.2f}h negative-eligible, {len(presses)} wake presses")
    for arm in results:
        print(f"  {arm}: {results[arm]}")

    if "A0_conv" in results and "A2_twin" in results:
        k, n = results["A0_conv"]["fa"], results["A0_conv"]["fa"] + results["A2_twin"]["fa"]
        if n > 0:
            r = binomtest(k, n, p=0.5, alternative="two-sided")
            print(f"\n  conventional vs response-matched (k={k}, n={n}): p={r.pvalue:.4f}")
        else:
            print("\n  n=0, no test possible")

    out = {"fixed_gain": FIXED_GAIN, "total_hours": round(total_hours, 3), "neg_hours": round(neg_hours, 3),
           "n_presses": len(presses), "default_det": default_det, "default_fa": default_fa,
           "target_det": target_det, "results": results}
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
