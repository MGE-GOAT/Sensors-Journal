'''Re-analyze the 10h fully-trained arms at MATCHED DETECTION (not fixed threshold).'''
import os, glob, json
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np, sys
sys.path.insert(0, os.path.expanduser('~/wuwexp'))
import eval_canonical as C, eval_extended_device16 as E
presses=[float(l) for l in open(os.path.join(E.SESSION_DIR,'button_presses.log')) if l.strip() and not l.startswith('#')]
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
    for _ in range(22):
        m=(lo+hi)/2;d,f=decide(p,m)
        if b is None or abs(d-tg)<abs(b[1]-tg):b=(m,d,f)
        if d==tg:return m,d,f
        elif d>tg:lo=m
        else:hi=m
        if hi-lo<0.002:break
    return b
preds={}
for f in sorted(glob.glob(os.path.expanduser('~/wuwexp/results/fulltrained/pr_*.npy'))):
    preds[os.path.basename(f)[3:-4]]=np.load(f)
out={}
for seed in ['s99','s100','s101']:
    arms={a:p for a,p in preds.items() if a.endswith(seed)}
    if len(arms)<4: continue
    d60={a:decide(p,0.60)[0] for a,p in arms.items()}
    tg=min(d60.values())
    print(f'--- {seed}: det@0.60={d60} target={tg}',flush=True)
    out[seed]={'target_det':int(tg),'arms':{}}
    for a,p in arms.items():
        th,d,f=mth(p,tg)
        out[seed]['arms'][a]={'th':round(float(th),3),'det':int(d),'fa':int(f)}
        print(f'   {a:22s} th={th:.3f} det={d} fa={f}',flush=True)
json.dump(out,open(os.path.expanduser('~/wuwexp/results/fulltrained/matched_10h.json'),'w'),indent=1)
print('MATCHED_DONE')
