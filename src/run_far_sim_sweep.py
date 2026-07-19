'''Threshold-SWEEP version of run_far_sim_all_arms.py.
Identical decision logic; inference runs ONCE per file, then the Schmitt/cooldown
state machine is replayed for a grid of THRESHOLD_UP values on the cached EMA trace.
Self-check: the sweep value at 0.60 must reproduce the published per-arm FA counts.
'''
import os, sys, glob, json, time
import numpy as np
import tensorflow as tf
from scipy.signal import lfilter

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
ALPHA = 0.3; THRESHOLD_DOWN = 0.30; MARGIN_MIN = 0.20; COOLDOWN_FRAMES = 7
GRID = [round(x,3) for x in np.arange(0.40, 0.901, 0.02)] + [0.60]
GRID = sorted(set(GRID))

def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=1); interp.allocate_tensors()
    i_d = interp.get_input_details(); o_d = interp.get_output_details()
    i_s, i_z = i_d[0]['quantization']; o_s, o_z = o_d[0]['quantization']
    return interp, i_d, o_d, i_s, i_z, o_s, o_z

def run_tflite(interp, i_d, o_d, i_s, i_z, o_s, o_z, feat):
    x = feat / i_s + i_z
    x = np.clip(x, -128, 127).astype(np.int8)
    interp.set_tensor(i_d[0]['index'], x); interp.invoke()
    out = interp.get_tensor(o_d[0]['index'])
    return (out.astype(np.float32) - o_z) * o_s

def fires_for_threshold(ema_wake, margin, th_up):
    cooldown = 0; armed = True; fires = 0
    for i in range(ema_wake.shape[0]):
        ew = ema_wake[i]
        if not armed and ew < THRESHOLD_DOWN: armed = True
        if cooldown > 0: cooldown -= 1
        if cooldown == 0 and armed and ew > th_up and margin[i] > MARGIN_MIN:
            fires += 1; armed = False; cooldown = COOLDOWN_FRAMES
    return fires

def main():
    arm = sys.argv[1]
    cache_files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz')))
    interp, i_d, o_d, i_s, i_z, o_s, o_z = load_tflite(ARMS[arm])
    fa = {t: 0 for t in GRID}; hours = 0.0; done = 0; failed = 0; t0 = time.time()
    for k, f in enumerate(cache_files):
        try:
            d = np.load(f); strides = d['features'].astype(np.float32); secs = float(d['audio_seconds'])
        except Exception:
            failed += 1; continue
        n = strides.shape[0]
        preds = np.empty((n,3), np.float32)
        for i in range(n):
            preds[i] = run_tflite(interp, i_d, o_d, i_s, i_z, o_s, o_z, strides[i:i+1])[0]
        # EMA is threshold-independent -> compute once (zero initial state, as in original)
        ema = lfilter([ALPHA], [1.0, -(1.0-ALPHA)], preds, axis=0).astype(np.float32)
        ema_wake = ema[:,0]; margin = ema_wake - np.maximum(ema[:,1], ema[:,2])
        for t in GRID:
            fa[t] += fires_for_threshold(ema_wake, margin, t)
        hours += secs/3600.0; done += 1
        if (k+1) % 20000 == 0:
            el=(time.time()-t0)/60
            print(f'[{arm}] {k+1}/{len(cache_files)} files {hours:.1f}h fa@0.60={fa[0.6]} {el:.1f}min', flush=True)
            json.dump({'arm':arm,'partial':True,'files':done,'hours':hours,'fa_by_threshold':{str(t):v for t,v in fa.items()}},
                      open(os.path.join(D, f'far_sweep_{arm}.json'),'w'), indent=1)
    out={'arm':arm,'partial':False,'files':done,'failed':failed,'hours':hours,
         'fa_by_threshold':{str(t):v for t,v in fa.items()},
         'decision_params':{'alpha':ALPHA,'threshold_down':THRESHOLD_DOWN,'margin_min':MARGIN_MIN,'cooldown':COOLDOWN_FRAMES},
         'elapsed_min':(time.time()-t0)/60}
    json.dump(out, open(os.path.join(D, f'far_sweep_{arm}.json'),'w'), indent=1)
    print(f'[{arm}] DONE fa@0.60={fa[0.6]} hours={hours:.1f}', flush=True)

main()
