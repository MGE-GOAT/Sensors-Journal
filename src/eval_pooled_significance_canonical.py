"""
Canonical (no-normalization) re-run of eval_pooled_significance.py.

Identical methodology to the original script -- same 7 positions (phase3
P1_0.3m/P2_1.5m/P3_4m/P4_otherroom_1.5m + phase3_room3 room3_0.3m/room3_1.5m/
room3_4m), same matched-detection-sensitivity threshold calibration (per-arm
threshold search so wake-stream detection count matches a shared target =
the minimum default-threshold (0.60) detection count across arms), same
exact two-sided binomial test on pooled A0_conv vs A2_twin false-alarm
counts -- EXCEPT audio loading and streaming feature extraction are now
imported unmodified from eval_canonical.py instead of this script's own
RMS-normalized load_ch0 (`a / rms * 0.05`), which does not match the
training/inference ground-truth recipe (see eval_canonical.py docstring).

count_acts()/find_matched_threshold() below reimplement the original
script's own grid-search threshold-matching logic (unchanged), but the
sequential EMA/Schmitt/margin/cooldown decision itself is delegated to
eval_canonical.run_decision so the decision logic is identical to what was
already verified against the published stream_P1_0.3m.json reference.
"""
import os, glob, json
import numpy as np
import tensorflow as tf
from scipy.stats import binomtest

from eval_canonical import load_ch0_raw, stream_feats, run_decision, STRIDE, SR

DEFAULT_TH_UP = 0.60

MODELS_DIR = os.path.expanduser("~/wuwexp/models/ladder")
PHASE3_DIR = os.path.expanduser("~/wuwexp/domainshift/phase3")
ROOM3_DIR = os.path.expanduser("~/wuwexp/domainshift/phase3_room3")


def count_acts(preds, th_up):
    return len(run_decision(preds, th_up))


def find_matched_threshold(wake_preds, target_det, th_grid):
    best_th, best_det = th_grid[0], count_acts(wake_preds, th_grid[0])
    for th in th_grid:
        det = count_acts(wake_preds, th)
        if det <= target_det:
            return th, det
        best_th, best_det = th, det
    return best_th, best_det


def main():
    all_positions = sorted(glob.glob(os.path.join(PHASE3_DIR, "P*"))) + \
                     sorted(glob.glob(os.path.join(ROOM3_DIR, "room3_*")))
    print(f"Pooling {len(all_positions)} positions across 3 rooms (CANONICAL, no-normalization pipeline):")
    for p in all_positions:
        print(f"  {p}")

    print("\nLoading models...")
    models = {}
    for mp in sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras"))):
        arm = os.path.splitext(os.path.basename(mp))[0]
        models[arm] = tf.keras.models.load_model(mp, compile=False)
    print(f"  arms: {list(models)}")

    th_grid = np.arange(0.50, 0.96, 0.001)
    per_position = {}
    total_fa = {arm: 0 for arm in models}
    total_neg_hours = 0.0

    for pos_dir in all_positions:
        pos = os.path.basename(pos_dir.rstrip("/"))
        print(f"\n--- {pos} ---")
        feats = {}
        for cls in ["wake", "speech", "noise"]:
            wav_path = os.path.join(pos_dir, f"{cls}.wav")
            if not os.path.exists(wav_path):
                print(f"  MISSING {wav_path}, skipping position")
                feats = None
                break
            a = load_ch0_raw(wav_path)
            feats[cls] = stream_feats(a)
        if feats is None:
            continue

        preds = {arm: {cls: m.predict(feats[cls], batch_size=512, verbose=0)
                       for cls in ["wake", "speech", "noise"]}
                  for arm, m in models.items()}

        default_det = {arm: count_acts(preds[arm]["wake"], DEFAULT_TH_UP) for arm in models}
        target_det = min(default_det.values())

        neg_hours = (len(feats["speech"]) + len(feats["noise"])) * (STRIDE / SR) / 3600.0
        total_neg_hours += neg_hours

        pos_result = {"default_det": default_det, "target_det": target_det, "neg_hours": round(neg_hours, 3), "arms": {}}
        for arm in models:
            th, det = find_matched_threshold(preds[arm]["wake"], target_det, th_grid)
            sfa = count_acts(preds[arm]["speech"], th)
            nfa = count_acts(preds[arm]["noise"], th)
            fa = sfa + nfa
            total_fa[arm] += fa
            pos_result["arms"][arm] = {"th": round(float(th), 3), "det": det, "fa": fa, "speech_fa": sfa, "noise_fa": nfa}
            print(f"  {arm:10s} th={th:.2f} det={det:3d} fa={fa:3d} (speech={sfa}, noise={nfa})")
        per_position[pos] = pos_result

    print(f"\n{'='*60}")
    print(f"POOLED across {len(per_position)} positions, {total_neg_hours:.2f}h negative audio total (CANONICAL pipeline):")
    for arm in sorted(total_fa):
        print(f"  {arm:12s} total FA = {total_fa[arm]}")

    pairs = [("A0_conv", "A2_twin", "conventional vs response-matched"),
             ("A0_conv", "A1_generic", "conventional vs microphone-agnostic control"),
             ("A2_twin", "A1_generic", "response-matched vs microphone-agnostic control"),
             ("A2_twin", "A3_randtwin", "response-matched vs randomized filter")]
    print()
    pvals = {}
    for a, b, label in pairs:
        if a not in total_fa or b not in total_fa:
            continue
        k, n = total_fa[a], total_fa[a] + total_fa[b]
        if n > 0:
            result = binomtest(k, n, p=0.5, alternative="two-sided")
            pvals[f"{a}_vs_{b}"] = result.pvalue
            print(f"  {label:50s} ({a}={total_fa[a]} vs {b}={total_fa[b]}, n={n}): p={result.pvalue:.4f}")
        else:
            pvals[f"{a}_vs_{b}"] = None
            print(f"  {label:50s} ({a}=0 vs {b}=0): no test possible, both zero")

    out = {"pipeline": "canonical_no_normalization",
           "per_position": per_position, "total_fa": total_fa,
           "total_neg_hours": round(total_neg_hours, 3),
           "pvalues": pvals}
    out_path = os.path.expanduser("~/wuwexp/domainshift/pooled_significance_results_CANONICAL.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
