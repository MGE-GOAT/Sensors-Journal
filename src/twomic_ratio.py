"""twomic_ratio.py -- INMP441 / PC-mic magnitude ratio from simultaneous pink-noise
recordings. The common phone-speaker + room cancels in the ratio, leaving the pure
mic-response difference -- exactly what the twin FIR is supposed to model.
Compares the measured ratio to the twin FIR, band by band."""
import os, numpy as np, wave
from scipy.signal import welch
from numpy.fft import rfft, rfftfreq
FS = 16000
os.chdir(os.path.expanduser("~/wuwexp/domainshift/twomic"))

def load(f, s32):
    w = wave.open(f, "rb"); ch = w.getnchannels(); n = w.getnframes()
    dt = np.int32 if s32 else np.int16
    a = np.frombuffer(w.readframes(n), dt).astype(np.float64).reshape(-1, ch)[:, 0]
    return a / (2.0**31 if s32 else 2.0**15)

pc = load("pc_ref.wav", False)     # PC GM303/default mic = conventional reference
inmp = load("inmp32.wav", True)    # INMP441 ch0

def ltas(x, nfft=4096):
    f, P = welch(x, FS, nperseg=nfft, noverlap=nfft//2); return f, np.sqrt(P + 1e-20)

fp, Mp = ltas(pc); fi, Mi = ltas(inmp)
Mi_i = np.interp(fp, fi, Mi)
ratio = Mi_i / (Mp + 1e-20)

twin = np.load("../twin_fir.npy")
fT = rfftfreq(4096, 1/FS); MT = np.abs(rfft(twin, 4096))

BANDS = [(100,300),(300,600),(600,1000),(1000,1500),(1500,2400),(2400,4000),(4000,7000)]
def bdb(f, M):
    v = [20*np.log10(M[(f>=lo)&(f<hi)].mean()+1e-20) for lo,hi in BANDS]
    return np.array([x - v[1] for x in v])           # normalize to 300-600 Hz band

rdb = bdb(fp, ratio); tdb = bdb(fT, MT)
print("band(Hz)   measured(INMP/PC)   twin_FIR    diff(dB)")
print("-"*52)
for k,(lo,hi) in enumerate(BANDS):
    print("%4d-%-4d     %7.1f          %7.1f     %6.1f" % (lo,hi,rdb[k],tdb[k],rdb[k]-tdb[k]))
print("-"*52)
print("shape correlation (measured vs twin): %.3f" % np.corrcoef(rdb, tdb)[0,1])
print("mean |diff| over bands: %.1f dB" % np.mean(np.abs(rdb-tdb)))
# also SNR sanity: how far is INMP above its own noise floor per band?
print("\nINMP band levels (dBFS, want well above ~-70 noise floor):")
for lo,hi in BANDS:
    b = 20*np.log10(Mi[(fi>=lo)&(fi<hi)].mean()+1e-20)
    print("  %4d-%-4d %6.1f dBFS" % (lo,hi,b))
