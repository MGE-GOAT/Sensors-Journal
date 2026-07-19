"""Export the 12 fully-trained ladder models to int8 tflite (calibrated on their OWN
training cache, clean/unaugmented), ready for eval_far.py over the negative corpus."""
import os, glob
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np, tensorflow as tf, sys
sys.path.insert(0, os.path.expanduser("~/wuwexp"))
import tflite_utils

CACHE = {"A0_conv":"features_pcen.npz", "A1_generic":"features_pcen_generic.npz",
         "A2_twin":"features_pcen_twin.npz", "A3_randtwin":"features_pcen_randtwin.npz"}
OUT = os.path.expanduser("~/wuwexp/models/fulltrained_tflite"); os.makedirs(OUT, exist_ok=True)
reps = {}
for mp in sorted(glob.glob(os.path.expanduser("~/wuwexp/models/fulltrained/*.keras"))):
    tag = os.path.splitext(os.path.basename(mp))[0]          # A2_twin_s99
    arm = "_".join(tag.split("_")[:-1])                       # A2_twin
    outp = os.path.join(OUT, tag + "_int8.tflite")
    if os.path.exists(outp):
        print("skip", tag, flush=True); continue
    if arm not in reps:
        reps[arm] = np.load(os.path.expanduser("~/wuwexp/cache/" + CACHE[arm]))["Xtr"]
    m = tf.keras.models.load_model(mp, compile=False)
    n = tflite_utils.to_int8_tflite(m, reps[arm], outp)
    print(f"exported {tag} -> {n/1024:.1f} KB", flush=True)
print("EXPORT_DONE")
