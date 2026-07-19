"""Score fully-trained arms (3 seeds each) on the 10h live session, matched detection."""
import os, glob, json
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np, tensorflow as tf, sys
sys.path.insert(0, os.path.expanduser("~/wuwexp"))
import eval_canonical as C, eval_extended_device16 as E
SESSION_DIR=E.SESSION_DIR
# preds cache from exp2 is ladder; recompute feats for fulltrained
wavs=sorted(glob.glob(os.path.join(SESSION_DIR,"live_extended*.wav")))
audio=E.load_ch0_concat_fixed_gain(wavs); feats=C.stream_feats(audio)
presses=[float(l) for l in open(os.path.join(SESSION_DIR,"button_presses.log")) if l.strip() and not l.startswith("#")]
ww=[(p-E.SESSION_START_EPOCH-E.WAKE_BEFORE_S,p-E.SESSION_START_EPOCH+E.WAKE_AFTER_S) for p in presses]
def decide(p,th):
    ema=np.zeros(p.shape[1],np.float32);cd,armed=0,True;det=fa=0
    for i in range(p.shape[0]):
        ema=C.ALPHA_EMA*p[i]+(1-C.ALPHA_EMA)*ema;ew=float(ema[0]);mg=ew-float(np.max(ema[1:]))
        if not armed and ew<C.TH_DOWN:armed=True
        if cd>0:cd-=1
        if cd==0 and armed and ew>th and mg>C.MARGIN_MIN:
            t=i*E.STRIDE/E.SR;w=any(lo<=t<=hi for lo,hi in ww);det+=w;fa+=(not w);armed=False;cd=C.COOLDOWN
    return det,fa
def mth(p,tg):
    lo,hi,b=0.50,0.95,None
    for _ in range(20):
        m=(lo+hi)/2;d,f=decide(p,m)
        if b is None or abs(d-tg)<abs(b[1]-tg):b=(m,d,f)
        if d==tg:return m
        elif d>tg:lo=m
        else:hi=m
        if hi-lo<0.002:break
    return b[0]
out={}
for mp in sorted(glob.glob(os.path.expanduser("~/wuwexp/models/fulltrained/*.keras"))):
    arm=os.path.splitext(os.path.basename(mp))[0]
    m=tf.keras.models.load_model(mp,compile=False)
    pr=m.predict(feats,batch_size=1024,verbose=0)
    d60,f60=decide(pr,0.60)
    out[arm]={"det@0.60":int(d60),"fa@0.60":int(f60),"_pr_saved":False}
    np.save(os.path.expanduser(f"~/wuwexp/results/fulltrained/pr_{arm}.npy"), pr.astype(np.float32))
    print(arm,out[arm],flush=True)
json.dump(out,open(os.path.expanduser("~/wuwexp/results/fulltrained/streaming_default.json"),"w"),indent=1)
print("FT_STREAM_DONE")
