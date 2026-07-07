"""phase4_eval.py -- score each ladder arm on the REAL-INMP441 recordings.
For a position dir (speech/noise/wake .wav, each = the played stream captured through
the INMP441), align to the played stream (cross-correlation on the first 30 s),
segment into the known clips (2.4 s clip + 0.3 s gap), compute the clip-level PCEN
feature, and score every saved arm model. Reports per-arm AUC / FAR@2%FRR /
per-source FA on the real microphone. Usage: python phase4_eval.py <position_dir>"""
import os, sys, json, glob, wave
import numpy as np
from scipy.signal import correlate
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import common as C

FS = 16000
CLIP = int(2.4 * FS); GAP = int(0.3 * FS); SPACING = CLIP + GAP
STREAMS = os.path.expanduser("~/wuwexp/domainshift/streams")
MODELS = os.path.expanduser("~/wuwexp/models/ladder")

def load_wav(p):
    w = wave.open(p, "rb"); n, ch, fs, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
    a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1: a = a.reshape(-1, ch)[:, 0]     # INMP441 on ch0
    return a, fs

def find_offset(rec, played):
    """lag (samples) of the played stream inside the recording, via xcorr on first 30 s."""
    a = rec[:30 * FS]; b = played[:min(len(played), 25 * FS)]
    xc = correlate(a - a.mean(), b - b.mean(), mode="valid")
    return int(np.argmax(np.abs(xc)))

def segment(rec, played, n_clips):
    off = find_offset(rec, played)
    clips = []
    for i in range(n_clips):
        s = off + i * SPACING
        c = rec[s:s + CLIP]
        if len(c) < CLIP: c = np.pad(c, (0, CLIP - len(c)))
        clips.append(c.astype(np.float32))
    return np.array(clips), off

def feats(clips):
    return np.stack([C.format_ds_cnn(C.extract_ds_cnn_mfe(c)) for c in clips]).astype(np.float32)

def far_at_frr(wake_p, is_wake, target=0.02):
    ths = np.unique(wake_p); best = 1.0
    for t in ths:
        fired = wake_p >= t
        frr = np.mean(~fired[is_wake]) if is_wake.any() else 0.0
        far = np.mean(fired[~is_wake]) if (~is_wake).any() else 0.0
        if frr <= target: best = min(best, far)
    return best

def main():
    pos_dir = os.path.expanduser(sys.argv[1]); pos = os.path.basename(pos_dir.rstrip("/"))
    man = json.load(open(f"{STREAMS}/manifest.json"))
    # segment each class's recording into per-clip audio
    X = {}; y = {}
    for cls, lab in [("wake", 0), ("speech", 1), ("noise", 2)]:
        rec, fs = load_wav(f"{pos_dir}/{cls}.wav")
        played, _ = load_wav(f"{STREAMS}/{cls}_stream.wav")
        n = man[cls]["n_clips"]
        clips, off = segment(rec, played, n)
        X[cls] = feats(clips); y[cls] = np.full(len(clips), lab)
        print("[seg] %-6s n=%d offset=%.2fs featmean=%.3f" % (cls, n, off / FS, float(X[cls].mean())), flush=True)
    Xall = np.concatenate([X["wake"], X["speech"], X["noise"]])
    yall = np.concatenate([y["wake"], y["speech"], y["noise"]])
    is_wake = (yall == 0)
    models = sorted(glob.glob(f"{MODELS}/*.keras"))
    if not models:
        print("NO MODELS YET (segmentation test only) — clips ready."); return
    rows = {}
    for mp in models:
        arm = os.path.splitext(os.path.basename(mp))[0]
        m = tf.keras.models.load_model(mp, compile=False)
        p = m.predict(Xall, batch_size=256, verbose=0)[:, 0]   # P(wake)
        auc = roc_auc_score(is_wake.astype(int), p)
        far2 = far_at_frr(p, is_wake)
        fired = p >= 0.5
        sfa = int(np.sum(fired[yall == 1])); nfa = int(np.sum(fired[yall == 2]))
        recall = float(np.mean(fired[is_wake]))
        rows[arm] = dict(auc=float(auc), far2=float(far2), speech_fa=sfa, noise_fa=nfa,
                         recall_at_0p5=recall, n_speech=int((yall == 1).sum()), n_noise=int((yall == 2).sum()))
        print("  %-12s AUC %.4f  FAR@2%%FRR %.3f%%  speechFA %d/%d  noiseFA %d/%d  recall@.5 %.2f"
              % (arm, auc, 100 * far2, sfa, (yall == 1).sum(), nfa, (yall == 2).sum(), recall), flush=True)
    out = os.path.expanduser(f"~/wuwexp/results/realmic_{pos}.json")
    json.dump(rows, open(out, "w"), indent=2); print("saved ->", out)

if __name__ == "__main__":
    main()
