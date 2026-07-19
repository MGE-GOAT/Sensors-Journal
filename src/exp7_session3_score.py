'''Score all 12 fully-trained arms on session-2 (pure negatives, no wake windows).
Uses session-1 matched thresholds -> every fire is a false alarm. Concatenates all
chunks, applies device-faithful x16 gain, streams the deployed decision logic.'''
import os, glob, json, wave
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np, tensorflow as tf, sys
sys.path.insert(0, os.path.expanduser('~/wuwexp'))
import eval_canonical as C

SESS=os.path.expanduser('~/wuwexp/session3')
FIXED_GAIN=16.0
# session-1 matched thresholds (from s99 fully-trained, the informative seed)
TH={'A0_conv':0.620,'A1_generic':0.584,'A2_twin':0.613,'A3_randtwin':0.598}

def load_ch0(path):
    w=wave.open(path,'rb'); n=w.getnframes(); raw=w.readframes(n)
    a=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    ch=w.getnchannels()
    if ch>1: a=a[::ch]  # channel 0
    return a

wavs=sorted(glob.glob(os.path.join(SESS,'*.wav')))
print(f'{len(wavs)} chunks',flush=True)
audio=np.concatenate([load_ch0(p) for p in wavs])
hours=len(audio)/C.SR/3600
audio=audio*FIXED_GAIN
print(f'total {hours:.2f}h, post-gain rms={np.sqrt(np.mean(audio**2)):.5f}',flush=True)
feats=C.stream_feats(audio)
print(f'{len(feats)} strides',flush=True)

def decide(p,th,dump=False):
    ema=np.zeros(p.shape[1],np.float32);cd,armed=0,True;fa=0;ft=[]
    for i in range(p.shape[0]):
        ema=C.ALPHA_EMA*p[i]+(1-C.ALPHA_EMA)*ema;ew=float(ema[0]);mg=ew-float(np.max(ema[1:]))
        if not armed and ew<C.TH_DOWN:armed=True
        if cd>0:cd-=1
        if cd==0 and armed and ew>th and mg>C.MARGIN_MIN:
            fa+=1
            if dump: ft.append(round(i*C.STRIDE/C.SR,1))
            armed=False;cd=C.COOLDOWN
    return fa,ft

out={'hours':round(hours,2),'arms':{}}
for mp in sorted(glob.glob(os.path.expanduser('~/wuwexp/models/fulltrained/*.keras'))):
    tag=os.path.splitext(os.path.basename(mp))[0]        # A2_twin_s99
    arm='_'.join(tag.split('_')[:-1])
    th=TH[arm]
    m=tf.keras.models.load_model(mp,compile=False)
    pr=m.predict(feats,batch_size=1024,verbose=0)
    fa,ft=decide(pr,th,dump=True)
    out['arms'][tag]={'th':th,'fa':fa,'fa_per_hr':round(fa/hours,4),'fa_times':ft}
    print(f'{tag:22s} th={th} fa={fa} ({fa/hours:.3f}/h)',flush=True)
json.dump(out,open(os.path.expanduser('~/wuwexp/results/fulltrained/session3_scored.json'),'w'),indent=1)
print('SESSION2_SCORED')
