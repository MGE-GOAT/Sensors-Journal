"""exp2_mcnemar_curves.py -- episode-level McNemar + FA-vs-det operating curves
for the 10h live session (DEVICE_GAIN16), ladder arms. CPU-only (training owns GPU).
Outputs: preds_10h_ladder.npz (cache), mcnemar_episodes.json, operating_curves.json"""
import os, glob, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, os.path.expanduser("~/wuwexp"))
import eval_canonical as C
import eval_extended_device16 as E

SESSION_DIR = E.SESSION_DIR
OUT_DIR = os.path.expanduser("~/wuwexp/results/exp2")
os.makedirs(OUT_DIR, exist_ok=True)
PREDS_CACHE = os.path.join(OUT_DIR, "preds_10h_ladder.npz")

# ---- presses / wake windows (identical to eval_extended_device16) ----
presses = []
with open(os.path.join(SESSION_DIR, "button_presses.log")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            presses.append(float(line))
wake_windows = [(p - E.SESSION_START_EPOCH - E.WAKE_BEFORE_S, p - E.SESSION_START_EPOCH + E.WAKE_AFTER_S) for p in presses]
total_wake_s = sum(hi - lo for lo, hi in wake_windows)

# ---- preds (cached) ----
if os.path.exists(PREDS_CACHE):
    z = np.load(PREDS_CACHE)
    preds = {k: z[k] for k in z.files if k != "_hours"}
    total_hours = float(z["_hours"])
    print(f"loaded cached preds: {list(preds)} hours={total_hours:.2f}")
else:
    wavs = sorted(glob.glob(os.path.join(SESSION_DIR, "live_extended*.wav")))
    audio = E.load_ch0_concat_fixed_gain(wavs)
    total_hours = len(audio) / E.SR / 3600
    feats = C.stream_feats(audio)
    print(f"feats: {len(feats)} strides")
    preds = {}
    for mp in sorted(glob.glob(os.path.expanduser("~/wuwexp/models/ladder/*.keras"))):
        arm = os.path.splitext(os.path.basename(mp))[0]
        m = tf.keras.models.load_model(mp, compile=False)
        preds[arm] = m.predict(feats, batch_size=1024, verbose=0).astype(np.float32)
        print(f"predicted {arm}")
    np.savez_compressed(PREDS_CACHE, _hours=np.array(total_hours), **preds)
neg_hours = total_hours - total_wake_s / 3600.0

def decide(p, th_up, fire_dump=False):
    ema = np.zeros(p.shape[1], dtype=np.float32); cd, armed = 0, True
    det, fa = 0, 0; fires = []
    for i in range(p.shape[0]):
        ema = C.ALPHA_EMA * p[i] + (1 - C.ALPHA_EMA) * ema
        ew = float(ema[0]); margin = ew - float(np.max(ema[1:]))
        if not armed and ew < C.TH_DOWN: armed = True
        if cd > 0: cd -= 1
        if cd == 0 and armed and ew > th_up and margin > C.MARGIN_MIN:
            t = i * E.STRIDE / E.SR
            w = any(lo <= t <= hi for lo, hi in wake_windows)
            det += w; fa += (not w)
            if fire_dump: fires.append((round(t,2), bool(w)))
            armed = False; cd = C.COOLDOWN
    return det, fa, fires

def matched_th(p, target):
    lo, hi, best = 0.50, 0.95, None
    for _ in range(20):
        mid = (lo + hi) / 2
        det, fa, _ = decide(p, mid)
        if best is None or abs(det - target) < abs(best[1] - target): best = (mid, det, fa)
        if det == target: return mid
        elif det > target: lo = mid
        else: hi = mid
        if hi - lo < 0.002: break
    return best[0]

# ---- matched thresholds (reproduce published protocol) ----
default_det = {a: decide(preds[a], E.DEFAULT_TH_UP)[0] for a in preds}
target = min(default_det.values())
ths = {a: matched_th(preds[a], target) for a in preds}
fires = {a: decide(preds[a], ths[a], fire_dump=True) for a in preds}
print("matched:", {a: (round(ths[a],3), fires[a][0], fires[a][1]) for a in preds})

# ---- episode clustering ----
def episodes(times, gap):
    if not times: return []
    eps = [[times[0]]]
    for t in times[1:]:
        if t - eps[-1][-1] <= gap: eps[-1].append(t)
        else: eps.append([t])
    return eps

fa_times = {a: [t for t, w in fires[a][2] if not w] for a in preds}
mc = {"matched_thresholds": {a: round(ths[a],3) for a in preds},
      "fa_times": fa_times, "neg_hours": round(neg_hours,3)}
for gap in (5.0, 10.0, 30.0):
    eps = episodes(sorted(fa_times["A0_conv"]), gap)
    # discordant episodes: A2_twin silent within [ep_start-gap, ep_end+gap]
    disc = 0
    for ep in eps:
        lo2, hi2 = ep[0]-gap, ep[-1]+gap
        if not any(lo2 <= t <= hi2 for t in fa_times["A2_twin"]): disc += 1
    mc[f"episodes_gap{int(gap)}s"] = {"k": len(eps), "discordant_vs_twin": disc,
                                       "p_sign": float(0.5)**disc if disc else 1.0}
print("episodes:", {k: v for k, v in mc.items() if k.startswith("episodes")})
json.dump(mc, open(os.path.join(OUT_DIR, "mcnemar_episodes.json"), "w"), indent=1)

# ---- operating curves ----
curves = {}
for a in preds:
    rows = []
    for th in np.arange(0.50, 0.951, 0.01):
        det, fa, _ = decide(preds[a], float(th))
        rows.append([round(float(th),2), det, fa])
    curves[a] = rows
    print(f"curve {a} done")
json.dump({"neg_hours": round(neg_hours,3), "curves": curves},
          open(os.path.join(OUT_DIR, "operating_curves.json"), "w"), indent=1)
print("EXP2_DONE")
