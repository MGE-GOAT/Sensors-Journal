"""build_twin_filter_v2.py -- SNR-aware fix for the conventional->INMP441 'twin'.

Same measurement (simultaneous sweep on both mics, ratio of PSDs = transfer
function), but masks out frequency bins where the conventional mic's own
signal is too close to its noise floor -- at those bins the sweep barely
registered, so INMP/conventional is dividing by near-nothing and produces a
spurious, unphysically large gain (found: +41 dB at 5.85 kHz, where the
conventional mic's PSD was 53 dB below its 1 kHz level). Masked bins are
excluded from the smoothing input entirely (not just clamped to unity) so
they can't pull down nearby well-measured bins' averages either.
"""
import os, numpy as np, wave
from scipy.signal import welch, resample_poly, firwin2, freqz, lfilter

def load(p):
    w = wave.open(p, "rb"); n, ch, fs, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(w.readframes(n), dtype=dt).astype(np.float64) / np.iinfo(dt).max
    if ch > 1: a = a.reshape(-1, ch)[:, 0]
    return a, fs

FS = 16000
D = os.path.expanduser("~/wuwexp/domainshift")
pc, fspc = load(f"{D}/pc_capture.wav")
inm, fsi = load(f"{D}/inmp_capture.wav")
pc16 = resample_poly(pc, FS, fspc)
sl = slice(int(2.0 * FS), int(10.5 * FS))
pcs, ins = pc16[sl], inm[sl]
nper = 2048
f, Ppc = welch(pcs, fs=FS, nperseg=nper)
_, Pin = welch(ins, fs=FS, nperseg=nper)

# --- SNR-aware mask: exclude bins where the conventional (source) mic's own
# signal is too far below its own peak level to trust the ratio there. ---
Ppc_db = 10 * np.log10(np.maximum(Ppc, 1e-20))
peak_db = Ppc_db[(f >= 60) & (f <= 7000)].max()
SNR_FLOOR_DB = 30.0  # bins more than this below the source's own peak are untrustworthy
trustworthy = Ppc_db >= (peak_db - SNR_FLOOR_DB)
n_excluded = int((~trustworthy & (f >= 60) & (f <= 7000)).sum())
print(f"[twin_v2] source peak={peak_db:.1f}dB, SNR floor={SNR_FLOOR_DB}dB -> "
      f"{n_excluded} bins excluded as untrustworthy (in 60-7000Hz band)")

Hmag = np.sqrt(np.maximum(Pin, 1e-20) / np.maximum(Ppc, 1e-20))

def smooth_masked(f, H, mask, frac=3):
    """1/3-octave smoothing that only averages over trustworthy bins; a
    bin with no trustworthy neighbors falls back to unity gain (no claim)."""
    out = np.empty_like(H)
    for i, fc in enumerate(f):
        if fc <= 0:
            out[i] = 1.0
            continue
        lo, hi = fc / 2 ** (1 / (2 * frac)), fc * 2 ** (1 / (2 * frac))
        m = (f >= lo) & (f <= hi) & mask
        out[i] = H[m].mean() if m.any() else 1.0
    return out

Hs = smooth_masked(f, Hmag, trustworthy)
Hs /= Hs[(f >= 200) & (f <= 500)].mean()

band = (f >= 60) & (f <= 7000)
gains = np.ones_like(Hs)
gains[band] = Hs[band]
# additionally: hard-clamp any surviving bin to a physically plausible range
# (a MEMS-vs-electret capsule difference should not exceed ~20 dB anywhere)
GAIN_CLAMP_DB = 20.0
gains_db = 20 * np.log10(np.maximum(gains, 1e-6))
n_clamped = int((np.abs(gains_db) > GAIN_CLAMP_DB).sum())
gains_db = np.clip(gains_db, -GAIN_CLAMP_DB, GAIN_CLAMP_DB)
gains = 10 ** (gains_db / 20)
print(f"[twin_v2] {n_clamped} bins clamped to +/-{GAIN_CLAMP_DB}dB as a physical-plausibility backstop")

fn = f / (FS / 2)
fn = np.clip(fn, 0, 1); fn[0], fn[-1] = 0.0, 1.0
NTAPS = 257
fir = firwin2(NTAPS, fn, gains)

OUT_PATH = f"{D}/twin_fir_v2.npy"
np.save(OUT_PATH, fir)
print(f"[twin_v2] saved to {OUT_PATH} (original twin_fir.npy left untouched)")

wv, hh = freqz(fir, worN=2048, fs=FS)
mag_db = 20 * np.log10(np.abs(hh) + 1e-12)
print(f"[twin_v2] full-range mag_db: min={mag_db.min():.2f} max={mag_db.max():.2f}  "
      f"peak-to-peak={mag_db.max()-mag_db.min():.2f} dB")
sb = (wv >= 300) & (wv <= 4000)
print(f"[twin_v2] speech band (300-4000Hz): min={mag_db[sb].min():.2f} max={mag_db[sb].max():.2f}  "
      f"swing={mag_db[sb].max()-mag_db[sb].min():.2f} dB")
idx_max = np.argmax(mag_db); idx_min = np.argmin(mag_db)
print(f"[twin_v2] global max at {wv[idx_max]:.0f}Hz = {mag_db[idx_max]:.2f}dB; "
      f"global min at {wv[idx_min]:.0f}Hz = {mag_db[idx_min]:.2f}dB")

# sanity: what happened specifically at the old spurious spike (5848 Hz)?
i5848 = np.argmin(np.abs(wv - 5848))
print(f"[twin_v2] gain at 5848Hz (was +41dB in the old filter): {mag_db[i5848]:.2f}dB now")
