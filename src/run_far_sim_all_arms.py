"""
Streaming false-alarm-rate simulation, replayed from cached PCEN features
(no re-extraction) against each arm's real TFLite model. Reuses the exact
decision logic (EMA smoothing, Schmitt trigger, margin, cooldown) from the
original notebook's cell 10, so results are directly comparable to what that
cell was designed to produce -- just far cheaper, since the expensive
streaming PCEN extraction happened once already and is cached to disk.

Runs one arm at a time (CPU-only, TFLite interpreter, cheap), writes each
arm's result to disk immediately (atomic write) so a crash on arm K doesn't
lose arms 1..K-1, mirroring tonight's crash-safety lesson from the training
launcher.
"""
import os
import sys
import glob
import json
import time
import tempfile

import numpy as np
import tensorflow as tf

# --- repro paths ---------------------------------------------------------
# The four arm models are committed under models/sweep_arms/. The 413.7 h
# stress-feature cache (per-file *.npz PCEN features derived from non-keyword
# speech + ESC-50 + VocalSound) is NOT distributed (8.7 GB); point
# STRESS_CACHE_DIR at your own copy to reproduce.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CACHE_DIR = os.environ.get('STRESS_CACHE_DIR', os.path.join(_ROOT, 'stress_feature_cache'))
_ARMDIR = os.path.join(_ROOT, 'models', 'sweep_arms')
ARMS = {
    'conventional':    os.path.join(_ARMDIR, 'conventional.tflite'),
    'eq_agnostic':     os.path.join(_ARMDIR, 'eq_agnostic.tflite'),
    'measured_filter': os.path.join(_ARMDIR, 'measured_filter.tflite'),
    'micaugment':      os.path.join(_ARMDIR, 'micaugment.tflite'),
}
# -------------------------------------------------------------------------

# Decision logic constants -- verbatim from the original notebook's cell 10.
ALPHA = 0.3
THRESHOLD_UP = 0.60
THRESHOLD_DOWN = 0.30
MARGIN_MIN = 0.20
COOLDOWN_FRAMES = 7

RESULT_PATH_TMPL = os.path.join(D, "far_sim_result_{arm}_{ts}.json")
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")


def atomic_write_json(path, obj):
    d = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_tflite(path):
    # num_threads=1: this script runs 4 arms as 4 CONCURRENT processes on
    # the same machine. Without an explicit cap, XNNPACK's default thread
    # count (often matching core count) causes severe oversubscription when
    # multiplied across 4 processes -- benchmarked ~5x slowdown vs a single
    # thread per process. One thread per process, 4 processes total, stays
    # well within the available cores with no cross-process contention.
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]
    return interp, input_details, output_details, input_scale, input_zero_point, output_scale, output_zero_point


def run_tflite(interp, input_details, output_details, input_scale, input_zero_point,
                output_scale, output_zero_point, features_float32):
    x = features_float32 / input_scale + input_zero_point
    x = np.clip(x, -128, 127).astype(np.int8)
    interp.set_tensor(input_details[0]["index"], x)
    interp.invoke()
    out = interp.get_tensor(output_details[0]["index"])
    return (out.astype(np.float32) - output_zero_point) * output_scale


def simulate_file(strides, interp, input_details, output_details,
                   input_scale, input_zero_point, output_scale, output_zero_point):
    """strides: (n_strides, TARGET_WIDTH, N_MELS, 1) float32, in original
    stream order. Returns number of false-alarm fires for this file."""
    ema_probs = np.zeros(3, dtype=np.float32)
    cooldown = 0
    schmitt_armed = True
    fires = 0

    for i in range(strides.shape[0]):
        feat = strides[i:i + 1]  # (1, TARGET_WIDTH, N_MELS, 1)
        preds = run_tflite(interp, input_details, output_details,
                            input_scale, input_zero_point, output_scale, output_zero_point, feat)[0]
        ema_probs = ALPHA * preds + (1.0 - ALPHA) * ema_probs

        ema_wake = float(ema_probs[0])
        margin = ema_wake - float(max(ema_probs[1], ema_probs[2]))

        if not schmitt_armed and ema_wake < THRESHOLD_DOWN:
            schmitt_armed = True

        if cooldown > 0:
            cooldown -= 1

        if (cooldown == 0 and schmitt_armed
                and ema_wake > THRESHOLD_UP
                and margin > MARGIN_MIN):
            fires += 1
            schmitt_armed = False
            cooldown = COOLDOWN_FRAMES

    return fires


def main():
    cache_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"Cached feature files: {len(cache_files)}")

    only_arm = sys.argv[1] if len(sys.argv) > 1 else None
    arms_to_run = {only_arm: ARMS[only_arm]} if only_arm else ARMS
    if only_arm and only_arm not in ARMS:
        print(f"FATAL: unknown arm {only_arm!r}, must be one of {list(ARMS)}", file=sys.stderr)
        sys.exit(1)

    for arm_name, tflite_path in arms_to_run.items():
        t0 = time.time()
        print(f"\n=== ARM: {arm_name} ===")
        if not os.path.exists(tflite_path):
            print(f"[{arm_name}] SKIP: tflite not found at {tflite_path}")
            continue

        (interp, input_details, output_details,
         input_scale, input_zero_point, output_scale, output_zero_point) = load_tflite(tflite_path)
        print(f"[{arm_name}] loaded {tflite_path}")
        print(f"[{arm_name}] input scale={input_scale:.6f} zp={input_zero_point}  "
              f"output scale={output_scale:.6f} zp={output_zero_point}")

        total_false_alarms = 0
        total_hours = 0.0
        n_files_done = 0
        n_files_failed = 0

        for i, f in enumerate(cache_files):
            try:
                d = np.load(f)
                strides = d["features"].astype(np.float32)
                audio_seconds = float(d["audio_seconds"])
            except Exception as e:
                n_files_failed += 1
                continue

            fires = simulate_file(strides, interp, input_details, output_details,
                                   input_scale, input_zero_point, output_scale, output_zero_point)
            total_false_alarms += fires
            total_hours += audio_seconds / 3600.0
            n_files_done += 1

            if (i + 1) % 20000 == 0:
                elapsed = time.time() - t0
                print(f"[{arm_name}] progress: {i+1}/{len(cache_files)} files, "
                      f"{total_hours:.1f}h simulated, {total_false_alarms} false alarms so far, "
                      f"elapsed={elapsed/60:.1f}min")

        elapsed = time.time() - t0
        far_per_hour = total_false_alarms / total_hours if total_hours > 0 else None
        result = {
            "arm_name": arm_name,
            "tflite_path": tflite_path,
            "n_files_simulated": n_files_done,
            "n_files_failed": n_files_failed,
            "total_hours": total_hours,
            "total_false_alarms": total_false_alarms,
            "far_per_hour": far_per_hour,
            "elapsed_minutes": elapsed / 60.0,
            "decision_params": {
                "alpha": ALPHA, "threshold_up": THRESHOLD_UP, "threshold_down": THRESHOLD_DOWN,
                "margin_min": MARGIN_MIN, "cooldown_frames": COOLDOWN_FRAMES,
            },
        }
        result_path = RESULT_PATH_TMPL.format(arm=arm_name, ts=TIMESTAMP)
        atomic_write_json(result_path, result)
        print(f"[{arm_name}] DONE: {total_false_alarms} false alarms over {total_hours:.2f}h "
              f"({far_per_hour:.4f}/hour), {n_files_failed} files failed to load, "
              f"elapsed={elapsed/60:.1f}min -> {result_path}")

    print("\nAll arms simulated.")


if __name__ == "__main__":
    main()
