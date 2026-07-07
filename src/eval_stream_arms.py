"""eval_stream_arms.py -- STREAMING false-alarm eval of the ladder arms on the real
INMP441 recorded streams (deployment-faithful: per-stride streaming PCEN + EMA +
Schmitt trigger + cooldown, exactly like eval_far.py / the device). Counts
activations over each recorded stream: wake-stream -> detections, speech/noise-stream
-> false alarms. Comparing arms on the SAME recording is a valid A/B (speaker+room
cancels). Usage: python eval_stream_arms.py <position_dir>"""
import os, sys, glob, json, wave
import numpy as np
import librosa
import tensorflow as tf

SR=16000; N_FFT=512; HOP=320; LOOKBACK=N_FFT-HOP; N_MELS=32; FMIN=60; FMAX=6000; TW=50
PCEN_ALPHA=0.90; PCEN_DELTA=2.0; PCEN_R=0.5; PCEN_TIME_C=0.10
ALPHA=0.3; TH_UP=0.60; TH_DOWN=0.30; MARGIN=0.20; COOLDOWN=7
STRIDE=int(0.3*SR); WIN=int(1.0*SR); MIN_LEN=int(2.4*SR)
MODELS=os.path.expanduser("~/wuwexp/models/ladder")

def load_ch0(path):
    w=wave.open(path,"rb"); sw=w.getsampwidth(); ch=w.getnchannels(); n=w.getnframes()
    dt={2:np.int16,4:np.int32}[sw]
    a=np.frombuffer(w.readframes(n),dt).astype(np.float32).reshape(-1,ch)[:,0]
    a=a/(2.0**(8*sw-1))
    a=a/(np.sqrt((a**2).mean())+1e-9)*0.05    # RMS-normalize to a speech-like level
    return a

def warm_zi(n=5):
    sil=np.zeros(WIN,dtype=np.float32)
    m=librosa.feature.melspectrogram(y=sil,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=1.0,center=False)
    zi=None
    for _ in range(n):
        _,zi=librosa.pcen(S=m*(2**31),sr=SR,hop_length=HOP,time_constant=PCEN_TIME_C,gain=PCEN_ALPHA,bias=PCEN_DELTA,power=PCEN_R,eps=1e-3,zi=zi,return_zf=True)
    return zi

def stream_feats(audio):
    """streaming PCEN -> array of (n_stride, 50, 32) features (sequential state)."""
    zi=warm_zi(); mel_buf=np.zeros((TW,N_MELS),dtype=np.float32); lookback=np.zeros(LOOKBACK,dtype=np.float32)
    out=[]
    for s in range(0,len(audio)-STRIDE+1,STRIDE):
        chunk=audio[s:s+STRIDE].astype(np.float32); inp=np.concatenate([lookback,chunk])
        mel=librosa.feature.melspectrogram(y=inp,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=1.0,center=False)
        lookback=chunk[-LOOKBACK:].astype(np.float32).copy() if len(chunk)>=LOOKBACK else np.concatenate([lookback,chunk])[-LOOKBACK:].astype(np.float32)
        mel,zi=librosa.pcen(S=mel*(2**31),sr=SR,hop_length=HOP,time_constant=PCEN_TIME_C,gain=PCEN_ALPHA,bias=PCEN_DELTA,power=PCEN_R,eps=1e-3,zi=zi,return_zf=True)
        nf=mel.T; n_new=nf.shape[0]
        if n_new>=TW: mel_buf=nf[-TW:].astype(np.float32)
        elif n_new>0: mel_buf=np.roll(mel_buf,-n_new,axis=0); mel_buf[-n_new:]=nf
        out.append(mel_buf.copy())
    return np.array(out,dtype=np.float32)

def count_acts(preds):
    """sequential EMA + Schmitt re-arm + margin + cooldown -> activation count."""
    nc=preds.shape[1]; ema=np.zeros(nc,dtype=np.float32); cd=0; armed=True; acts=0
    for p in preds:
        ema=ALPHA*p+(1-ALPHA)*ema; ew=float(ema[0]); margin=ew-float(np.max(ema[1:]))
        if not armed and ew<TH_DOWN: armed=True
        if cd>0: cd-=1
        if cd==0 and armed and ew>TH_UP and margin>MARGIN:
            acts+=1; armed=False; cd=COOLDOWN
    return acts

def main():
    pos_dir=os.path.expanduser(sys.argv[1]); pos=os.path.basename(pos_dir.rstrip("/"))
    man=json.load(open(os.path.expanduser("~/wuwexp/domainshift/streams/manifest.json")))
    # precompute streaming features per class (shared across arms)
    feats={}; secs={}
    for cls in ["wake","speech","noise"]:
        a=load_ch0(f"{pos_dir}/{cls}.wav"); feats[cls]=stream_feats(a); secs[cls]=len(a)/SR
        print("[feat] %-6s strides=%d dur=%.0fs"%(cls,len(feats[cls]),secs[cls]),flush=True)
    neg_h=(secs["speech"]+secs["noise"])/3600.0
    print("\n%-12s | wakeDET/%d | speechFA noiseFA | FA/hr" % ("arm", man["wake"]["n_clips"]))
    print("-"*60); rows={}
    for mp in sorted(glob.glob(f"{MODELS}/*.keras")):
        arm=os.path.splitext(os.path.basename(mp))[0]; m=tf.keras.models.load_model(mp,compile=False)
        det=count_acts(m.predict(feats["wake"],batch_size=512,verbose=0))
        sfa=count_acts(m.predict(feats["speech"],batch_size=512,verbose=0))
        nfa=count_acts(m.predict(feats["noise"],batch_size=512,verbose=0))
        fahr=(sfa+nfa)/neg_h if neg_h>0 else 0
        rows[arm]=dict(det=det,speech_fa=sfa,noise_fa=nfa,fa_per_hr=round(fahr,1))
        print("%-12s |   %3d     |   %3d      %3d   | %6.1f"%(arm,det,sfa,nfa,fahr),flush=True)
    json.dump(rows,open(os.path.expanduser(f"~/wuwexp/results/stream_{pos}.json"),"w"),indent=2)

if __name__=="__main__": main()
