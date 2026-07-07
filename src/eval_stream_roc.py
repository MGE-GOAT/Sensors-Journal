"""eval_stream_roc.py -- streaming FA-vs-detection tradeoff across ALL positions.
Per arm, sweep the Schmitt up-threshold; at each, count wake detections and
speech+noise false alarms (deployment-faithful streaming), summed over P1-P4.
Then report each arm's FA at MATCHED detection -- the fair streaming A/B."""
import os, glob, json, wave
import numpy as np, librosa
import tensorflow as tf

SR=16000; N_FFT=512; HOP=320; LOOKBACK=N_FFT-HOP; N_MELS=32; FMIN=60; FMAX=6000; TW=50
PCEN_ALPHA=0.90; PCEN_DELTA=2.0; PCEN_R=0.5; PCEN_TIME_C=0.10
ALPHA=0.3; TH_DOWN_FRAC=0.5; MARGIN=0.20; COOLDOWN=7
STRIDE=int(0.3*SR)
MODELS=os.path.expanduser("~/wuwexp/models/ladder")
POSITIONS=["P1_0.3m","P2_1.5m","P3_4m","P4_otherroom_1.5m"]
THS=[0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]

def load_ch0(path):
    w=wave.open(path,"rb"); sw=w.getsampwidth(); ch=w.getnchannels()
    a=np.frombuffer(w.readframes(w.getnframes()),{2:np.int16,4:np.int32}[sw]).astype(np.float32).reshape(-1,ch)[:,0]
    a=a/(2.0**(8*sw-1)); return a/(np.sqrt((a**2).mean())+1e-9)*0.05

def warm_zi(n=5):
    sil=np.zeros(SR,dtype=np.float32); m=librosa.feature.melspectrogram(y=sil,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=1.0,center=False); zi=None
    for _ in range(n): _,zi=librosa.pcen(S=m*(2**31),sr=SR,hop_length=HOP,time_constant=PCEN_TIME_C,gain=PCEN_ALPHA,bias=PCEN_DELTA,power=PCEN_R,eps=1e-3,zi=zi,return_zf=True)
    return zi

def stream_feats(a):
    zi=warm_zi(); mel_buf=np.zeros((TW,N_MELS),dtype=np.float32); lb=np.zeros(LOOKBACK,dtype=np.float32); out=[]
    for s in range(0,len(a)-STRIDE+1,STRIDE):
        ch=a[s:s+STRIDE].astype(np.float32); inp=np.concatenate([lb,ch])
        mel=librosa.feature.melspectrogram(y=inp,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=1.0,center=False)
        lb=ch[-LOOKBACK:].astype(np.float32).copy() if len(ch)>=LOOKBACK else np.concatenate([lb,ch])[-LOOKBACK:].astype(np.float32)
        mel,zi=librosa.pcen(S=mel*(2**31),sr=SR,hop_length=HOP,time_constant=PCEN_TIME_C,gain=PCEN_ALPHA,bias=PCEN_DELTA,power=PCEN_R,eps=1e-3,zi=zi,return_zf=True)
        nf=mel.T; nn=nf.shape[0]
        if nn>=TW: mel_buf=nf[-TW:].astype(np.float32)
        elif nn>0: mel_buf=np.roll(mel_buf,-nn,axis=0); mel_buf[-nn:]=nf
        out.append(mel_buf.copy())
    return np.array(out,dtype=np.float32)

def count_acts(preds,th_up):
    th_dn=th_up*TH_DOWN_FRAC; ema=np.zeros(preds.shape[1],dtype=np.float32); cd=0; armed=True; acts=0
    for p in preds:
        ema=ALPHA*p+(1-ALPHA)*ema; ew=float(ema[0]); mg=ew-float(np.max(ema[1:]))
        if not armed and ew<th_dn: armed=True
        if cd>0: cd-=1
        if cd==0 and armed and ew>th_up and mg>MARGIN: acts+=1; armed=False; cd=COOLDOWN
    return acts

# precompute streaming features once per (position,class)
feats={}
for pos in POSITIONS:
    for cls in ["wake","speech","noise"]:
        feats[(pos,cls)]=stream_feats(load_ch0(os.path.expanduser(f"~/wuwexp/domainshift/phase3/{pos}/{cls}.wav")))
    print("[feat] %s done"%pos,flush=True)

arms=sorted(glob.glob(f"{MODELS}/*.keras"))
curves={}
for mp in arms:
    arm=os.path.splitext(os.path.basename(mp))[0]; m=tf.keras.models.load_model(mp,compile=False)
    pred={k:m.predict(v,batch_size=512,verbose=0) for k,v in feats.items()}
    curve=[]
    for th in THS:
        det=sum(count_acts(pred[(p,"wake")],th) for p in POSITIONS)
        fa=sum(count_acts(pred[(p,"speech")],th)+count_acts(pred[(p,"noise")],th) for p in POSITIONS)
        curve.append((th,det,fa))
    curves[arm]=curve
    print("\n%s (det out of 400 wake clips, FA over ~1.8h neg):"%arm,flush=True)
    for th,det,fa in curve: print("  th=%.2f  det=%3d  FA=%3d"%(th,det,fa),flush=True)

# fair comparison: FA at matched detection (interpolate each arm's FA at target det levels)
print("\n==== FA at MATCHED detection (lower FA = better) ====",flush=True)
for target in [80,120,160,200]:
    row=[]
    for arm,curve in curves.items():
        ds=[d for _,d,_ in curve]; fs=[f for _,_,f in curve]
        order=np.argsort(ds); d_s=np.array(ds)[order]; f_s=np.array(fs)[order]
        fa=float(np.interp(target,d_s,f_s)) if d_s.max()>=target>=d_s.min() else float("nan")
        row.append("%s=%.0f"%(arm.split("_")[0],fa))
    print("  det=%3d/400: %s"%(target,"  ".join(row)),flush=True)
json.dump({a:c for a,c in curves.items()},open(os.path.expanduser("~/wuwexp/results/stream_roc.json"),"w"),indent=2)
