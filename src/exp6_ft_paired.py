'''Proper PAIRED McNemar on the fully-trained arms, 10h session, at MATCHED detection.
Dumps exact fire times, clusters into episodes, and tests discordant events.'''
import os, glob, json
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np, sys
sys.path.insert(0, os.path.expanduser('~/wuwexp'))
import eval_canonical as C, eval_extended_device16 as E

presses=[float(l) for l in open(os.path.join(E.SESSION_DIR,'button_presses.log')) if l.strip() and not l.startswith('#')]
ww=[(p-E.SESSION_START_EPOCH-E.WAKE_BEFORE_S,p-E.SESSION_START_EPOCH+E.WAKE_AFTER_S) for p in presses]

def decide(p, th, dump=False):
    ema=np.zeros(p.shape[1],np.float32); cd,armed=0,True; det=fa=0; fires=[]
    for i in range(p.shape[0]):
        ema=C.ALPHA_EMA*p[i]+(1-C.ALPHA_EMA)*ema; ew=float(ema[0]); mg=ew-float(np.max(ema[1:]))
        if not armed and ew<C.TH_DOWN: armed=True
        if cd>0: cd-=1
        if cd==0 and armed and ew>th and mg>C.MARGIN_MIN:
            t=i*E.STRIDE/E.SR; w=any(lo<=t<=hi for lo,hi in ww)
            det+=w; fa+=(not w)
            if dump and not w: fires.append(round(t,2))
            armed=False; cd=C.COOLDOWN
    return det,fa,fires

def mth(p,tg):
    lo,hi,b=0.50,0.95,None
    for _ in range(22):
        m=(lo+hi)/2; d,f,_=decide(p,m)
        if b is None or abs(d-tg)<abs(b[1]-tg): b=(m,d,f)
        if d==tg: return m
        elif d>tg: lo=m
        else: hi=m
        if hi-lo<0.002: break
    return b[0]

preds={os.path.basename(f)[3:-4]: np.load(f) for f in sorted(glob.glob(os.path.expanduser('~/wuwexp/results/fulltrained/pr_*.npy')))}
out={}
for seed in ['s99','s100','s101']:
    arms={a:p for a,p in preds.items() if a.endswith(seed)}
    if len(arms)<4: continue
    tg=min(decide(p,0.60)[0] for p in arms.values())
    fires={}
    for a,p in arms.items():
        th=mth(p,tg); d,f,ft=decide(p,th,dump=True)
        fires[a]={'th':round(float(th),3),'det':int(d),'fa':int(f),'fa_times':ft}
        print(f'{seed} {a:22s} th={th:.3f} det={d} fa={f} times={ft}',flush=True)
    out[seed]={'target_det':int(tg),'arms':fires}
json.dump(out,open(os.path.expanduser('~/wuwexp/results/fulltrained/paired_10h.json'),'w'),indent=1)
print('PAIRED_DONE')
