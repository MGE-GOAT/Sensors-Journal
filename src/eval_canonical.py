"""
eval_canonical.py -- the CANONICAL, ground-truth-faithful feature extraction +
streaming decision module for ladder-arm evaluation.

Ground truth is /home/mahrad/storage/Data/wake/WUW-final/Wake Up Word/Ds-CNN-QAT.ipynb
(cells 0, 1, 2, 19), read verbatim. That notebook does:
  - librosa.load(path, sr=16000, mono=True) with ZERO amplitude normalization
    (no RMS rescale, no peak normalize -- just pad/truncate). The NORMALIZE=
    'per_sample' config constant in cell 0 is declared but never referenced by
    the actual feature-extraction code (cell 2) -- it is dead/vestigial.
  - raw audio -> librosa.feature.melspectrogram(power=1.0) -> librosa.pcen(
    S=mel*(2**31), ...) -> ring buffer of the last 50 frames, with PCEN
    adaptive state (zi) carried CONTINUOUSLY across strides (not reset per
    call) and warmed with 5 passes of silence before first use. Cell 2's
    comment: "training and inference are coupled. Both must use this recipe."

DO NOT add any raw-audio RMS/peak normalization to this file. An earlier
script tonight (eval_stream_arms.py, the one that produced the numbers
currently sitting in the Sensors Journal paper's tab:twin table) contains
the line:
    a = a / (np.sqrt((a ** 2).mean()) + 1e-9) * 0.05
which rescales every clip to a fixed RMS target of 0.05 before feature
extraction. That line does NOT appear anywhere in the notebook ground truth
above and was copied, uncommented, into every eval_*.py script written
tonight (eval_pooled_significance.py, eval_extended_session.py,
eval_stream_roc.py, eval_stream_roc2.py, eval_room3_matched.py,
eval_room3_matched_v2.py). This module is the corrected replacement: audio
loading here does bit-depth-correct PCM->float conversion only, never a
loudness rescale.
"""
import os
import numpy as np
import librosa

# ============================ Constants (notebook cell 0) ============================
SR           = 16000
N_FFT        = 512
HOP          = 320
LOOKBACK     = N_FFT - HOP        # 192
N_MELS       = 32
FMIN         = 60
FMAX         = 6000
TW           = 50                 # TARGET_WIDTH -- frames kept in the ring buffer

PCEN_ALPHA   = 0.90               # PCEN "gain"
PCEN_DELTA   = 2.0                # PCEN "bias"
PCEN_R       = 0.5                # PCEN "power"
PCEN_TIME_C  = 0.10                # PCEN "time_constant"
PCEN_SCALE   = 2 ** 31            # mel is scaled by this before librosa.pcen (cell 2/19)
PCEN_EPS     = 1e-3
PCEN_WARMUP_PASSES = 5

STRIDE       = int(0.3 * SR)      # 4800 samples = 0.3s (STRIDE_LEN in the notebook)

# ---- streaming decision logic defaults (matches eval_far.py / eval_stream_arms.py /
#      the device firmware; also the values the task's ground-truth notes specify) ----
ALPHA_EMA    = 0.3
TH_UP        = 0.60
TH_DOWN      = 0.30
MARGIN_MIN   = 0.20
COOLDOWN     = 7


# ============================ Audio loading ============================
def load_ch0_raw(path, channels=2, sampwidth=2):
    """Reads raw PCM directly from `path`, skipping the standard 44-byte
    canonical WAV header and trusting the file's ACTUAL size on disk rather
    than the header's declared data-size field, then returns mono channel 0
    as float32 in [-1, 1].

    This is required for segment 1 of the 10h live extended session: its PCM
    payload is exactly 2**31 bytes -- one byte over what a standard WAV
    header's 32-bit signed size field can represent, so the header's declared
    size silently overflows and Python's `wave` module (`wave.open` /
    `getnframes` / `readframes`) misreports the frame count on that specific
    file. Reading raw PCM from actual file size sidesteps the corrupted
    header entirely. This is a real, already-diagnosed, and unrelated issue
    from the RMS-normalization bug -- it is preserved here because it is
    correct, not because it is suspect.

    NOTE: only the bit-depth PCM->float conversion happens here
    (a / 2**(8*sampwidth-1)). There is deliberately NO RMS/peak
    normalization anywhere in this function -- the notebook ground truth
    (cell 1, load_audio) never rescales loudness, it only pads/truncates.
    """
    file_size = os.path.getsize(path)
    header_size = 44  # standard canonical WAV header (RIFF/fmt/data chunk headers only)
    with open(path, "rb") as f:
        f.seek(header_size)
        raw = f.read(file_size - header_size)

    frame_bytes = channels * sampwidth
    usable_len = (len(raw) // frame_bytes) * frame_bytes
    if usable_len != len(raw):
        print(f"  WARNING: {path} trailing {len(raw) - usable_len} bytes dropped (partial frame)")

    dtype = {2: np.int16, 4: np.int32}[sampwidth]
    a = np.frombuffer(raw[:usable_len], dtype).astype(np.float32).reshape(-1, channels)[:, 0]
    a = a / (2.0 ** (8 * sampwidth - 1))   # PCM int -> float32 in [-1, 1]; NOT a loudness rescale
    return a


# ============================ Streaming PCEN features (notebook cells 2 / 19) ============================
def _warm_pcen_zi(n_warmup=PCEN_WARMUP_PASSES):
    """5 passes of silence through PCEN to get a stable initial adaptive
    state, exactly as the notebook's _make_pcen_zi_init does."""
    silence = np.zeros(SR, dtype=np.float32)  # 1s of silence
    mel_sil = librosa.feature.melspectrogram(
        y=silence, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0, center=False)
    zi = None
    for _ in range(n_warmup):
        _, zi = librosa.pcen(
            S=mel_sil * PCEN_SCALE, sr=SR, hop_length=HOP,
            time_constant=PCEN_TIME_C, gain=PCEN_ALPHA, bias=PCEN_DELTA,
            power=PCEN_R, eps=PCEN_EPS, zi=zi, return_zf=True)
    return zi


def stream_feats(audio):
    """Streams `audio` (an arbitrarily long 1-D float array, raw amplitude,
    no normalization) through mel + PCEN feature extraction with ONE
    continuous PCEN adaptive state (zi) across the entire array -- state is
    never reset mid-stream, matching genuine always-on deployment and the
    notebook's cell-19 live-inference recipe.

    Processes the array in STRIDE (0.3s / 4800-sample) chunks, carrying a
    LOOKBACK (192-sample) tail between chunks, same as the notebook's
    _stream_pcen_into_buffer. The final chunk may be shorter than STRIDE
    (the notebook's `while cursor < n` loop processes it too, it does not
    truncate/drop it) -- handled here the same way.

    Returns an array of shape (n_strides, TW, N_MELS): one (50, 32) ring-
    buffer snapshot per stride, suitable for feeding a Keras model with
    model.predict(feats) to get one prediction per stride for run_decision().
    """
    zi = _warm_pcen_zi()
    mel_buffer = np.zeros((TW, N_MELS), dtype=np.float32)
    lookback = np.zeros(LOOKBACK, dtype=np.float32)
    out = []

    cursor = 0
    n = len(audio)
    while cursor < n:
        end = min(cursor + STRIDE, n)
        chunk = audio[cursor:end].astype(np.float32)
        cursor = end

        input_audio = np.concatenate([lookback, chunk])
        if len(input_audio) < N_FFT:
            break  # too short a tail to compute even one FFT frame

        mel = librosa.feature.melspectrogram(
            y=input_audio, sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0, center=False)

        if len(chunk) >= LOOKBACK:
            lookback = chunk[-LOOKBACK:].copy()
        else:
            lookback = np.concatenate([lookback, chunk])[-LOOKBACK:]

        mel, zi = librosa.pcen(
            S=mel * PCEN_SCALE, sr=SR, hop_length=HOP,
            time_constant=PCEN_TIME_C, gain=PCEN_ALPHA, bias=PCEN_DELTA,
            power=PCEN_R, eps=PCEN_EPS, zi=zi, return_zf=True)

        new_frames = mel.T
        n_new = new_frames.shape[0]
        if n_new == 0:
            continue
        if n_new >= TW:
            mel_buffer = new_frames[-TW:].astype(np.float32)
        else:
            mel_buffer = np.roll(mel_buffer, -n_new, axis=0)
            mel_buffer[-n_new:] = new_frames

        out.append(mel_buffer.copy())

    return np.array(out, dtype=np.float32)


# ============================ Streaming decision logic ============================
def run_decision(preds, th_up, ema_alpha=ALPHA_EMA, th_down=TH_DOWN,
                  margin_min=MARGIN_MIN, cooldown=COOLDOWN):
    """Sequential EMA + Schmitt-trigger re-arm + margin + cooldown decision
    logic (matches eval_far.py / eval_stream_arms.py / the device firmware).

    preds: (n_strides, n_classes) softmax outputs, one row per stream stride
    (e.g. from a Keras model's model.predict(stream_feats(audio))). Class 0
    is assumed to be the wake-word class.

    Returns a sorted list of stride indices at which the detector fired
    (each index i corresponds to time i * STRIDE / SR seconds into the
    stream that `preds` was computed from).
    """
    preds = np.asarray(preds, dtype=np.float32)
    n_classes = preds.shape[1]
    ema = np.zeros(n_classes, dtype=np.float32)
    cooldown_remaining = 0
    armed = True
    fires = []

    for i in range(preds.shape[0]):
        p = preds[i]
        ema = ema_alpha * p + (1 - ema_alpha) * ema
        wake_ema = float(ema[0])
        margin = wake_ema - float(np.max(ema[1:])) if n_classes > 1 else wake_ema

        if not armed and wake_ema < th_down:
            armed = True
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        if cooldown_remaining == 0 and armed and wake_ema > th_up and margin > margin_min:
            fires.append(i)
            armed = False
            cooldown_remaining = cooldown

    return fires


def find_matched_threshold(preds, target_det, is_wake_fn, lo=0.50, hi=0.95,
                            precision=0.002, max_iter=20):
    """Binary-search threshold calibration: finds a th_up such that
    run_decision(preds, th_up) produces a detection count as close as
    possible to `target_det`. Valid because detection count is monotonically
    non-increasing as th_up rises (higher threshold => fewer or equal
    fires) -- this is the matched-detection-sensitivity methodology used
    across tonight's evals, which avoids comparing arms at different
    operating points (the FRR confound already flagged/fixed for
    tab:speakerdisjoint in the paper). It is ~10x cheaper than a linear
    threshold scan at the same precision.

    preds: (n_strides, n_classes) softmax outputs for ONE arm's model.
    target_det: desired number of true-positive (wake-window) fires.
    is_wake_fn: callable(t_seconds) -> bool. Given the time in seconds of a
        fire (stride_index * STRIDE / SR into the stream `preds` was
        computed from), returns True if that fire falls inside a wake
        window (true positive) and False otherwise (false alarm).
    lo, hi: initial threshold search bracket.
    precision: stop once the bracket [lo, hi] narrows below this.
    max_iter: hard cap on binary-search iterations.

    Returns (threshold, det, fa) for the best threshold found (closest
    det to target_det; exact match returns immediately).
    """
    def _det_fa_at(th_up):
        fires = run_decision(preds, th_up)
        det = 0
        fa = 0
        for i in fires:
            t = i * STRIDE / SR
            if is_wake_fn(t):
                det += 1
            else:
                fa += 1
        return det, fa

    best = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        det, fa = _det_fa_at(mid)

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


# ============================ Smoke test ============================
if __name__ == "__main__":
    import wave
    import tempfile

    print("[smoke] building tiny synthetic WAV (2ch, int16, 3s @ 16kHz)...")
    rng = np.random.default_rng(0)
    n_samples = 3 * SR
    ch0 = (rng.standard_normal(n_samples) * 3000).astype(np.int16)
    ch1 = (rng.standard_normal(n_samples) * 3000).astype(np.int16)
    interleaved = np.empty(n_samples * 2, dtype=np.int16)
    interleaved[0::2] = ch0
    interleaved[1::2] = ch1

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_path = tf.name
    with wave.open(tmp_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(interleaved.tobytes())

    print("[smoke] load_ch0_raw(...)")
    audio = load_ch0_raw(tmp_path, channels=2, sampwidth=2)
    os.remove(tmp_path)
    assert audio.ndim == 1 and len(audio) > 0, "load_ch0_raw returned empty/bad shape"
    assert audio.min() >= -1.0 and audio.max() <= 1.0, "load_ch0_raw not in [-1, 1]"
    print(f"  audio: shape={audio.shape} dtype={audio.dtype} "
          f"min={audio.min():.4f} max={audio.max():.4f}")

    print("[smoke] stream_feats(...)")
    feats = stream_feats(audio)
    assert feats.ndim == 3 and feats.shape[1] == TW and feats.shape[2] == N_MELS, \
        f"stream_feats bad shape: {feats.shape}"
    print(f"  feats: shape={feats.shape}")

    print("[smoke] run_decision(...) on synthetic softmax-like preds")
    n_strides = feats.shape[0]
    n_classes = 3
    raw_preds = rng.random((n_strides, n_classes)).astype(np.float32)
    preds = raw_preds / raw_preds.sum(axis=1, keepdims=True)  # fake softmax rows
    fires_default = run_decision(preds, TH_UP)
    print(f"  n_strides={n_strides} fires@TH_UP={TH_UP}: {fires_default}")
    fires_low_th = run_decision(preds, 0.05)
    assert len(fires_low_th) >= len(fires_default), \
        "lower threshold should not produce fewer fires"

    print("[smoke] find_matched_threshold(...)")
    def is_wake_fn(t_seconds):
        return t_seconds < 1.0  # pretend the first second is a "wake window"
    target_det = len(fires_default)
    th, det, fa = find_matched_threshold(preds, target_det, is_wake_fn)
    print(f"  matched threshold={th:.4f} det={det} fa={fa} (target_det={target_det})")

    print("\n[smoke] ALL CHECKS PASSED")
