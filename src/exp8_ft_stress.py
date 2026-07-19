'''Run ONE fully-trained ladder tflite over the 413.7h stress cache at a fixed
(session-1 matched) threshold. Reports total false alarms + FA/h. CPU, num_threads=1.'''
import os, sys, glob, json, time
import numpy as np, tensorflow as tf
from scipy.signal import lfilter
D='/home/mahrad/storage/Data/wake/WUW-final/Wake Up Word/1'
CACHE=os.path.join(D,'stress_feature_cache')
ALPHA=0.3; TH_DOWN=0.30; MARGIN_MIN=0.20; COOLDOWN=7
# session-1 matched thresholds by arm
TH={'A0_conv':0.620,'A1_generic':0.584,'A2_twin':0.613,'A3_randtwin':0.598}
tag=sys.argv[1]                 # e.g. A2_twin_s99
arm='_'.join(tag.split('_')[:-1]); TH_UP=TH[arm]
mp=os.path.expanduser(f'~/wuwexp/models/fulltrained_tflite/{tag}_int8.tflite')
it=tf.lite.Interpreter(model_path=mp,num_threads=1); it.allocate_tensors()
ind=it.get_input_details(); outd=it.get_output_details()
i_s,i_z=ind[0]['quantization']; o_s,o_z=outd[0]['quantization']
def run(feat):
    x=np.clip(feat/i_s+i_z,-128,127).astype(np.int8)
    it.set_tensor(ind[0]['index'],x); it.invoke()
    return (it.get_tensor(outd[0]['index']).astype(np.float32)-o_z)*o_s
def fires(ema_w,mg):
    cd=0;armed=True;f=0
    for i in range(len(ema_w)):
        if not armed and ema_w[i]<TH_DOWN:armed=True
        if cd>0:cd-=1
        if cd==0 and armed and ema_w[i]>TH_UP and mg[i]>MARGIN_MIN:
            f+=1;armed=False;cd=COOLDOWN
    return f
files=sorted(glob.glob(os.path.join(CACHE,'*.npz')))
fa=0;hours=0.0;done=0;t0=time.time()
for k,fp in enumerate(files):
    try:
        d=np.load(fp);st=d['features'].astype(np.float32);sec=float(d['audio_seconds'])
    except Exception:continue
    preds=np.empty((st.shape[0],3),np.float32)
    for i in range(st.shape[0]):preds[i]=run(st[i:i+1])[0]
    ema=lfilter([ALPHA],[1.0,-(1.0-ALPHA)],preds,axis=0).astype(np.float32)
    fa+=fires(ema[:,0],ema[:,0]-np.maximum(ema[:,1],ema[:,2]));hours+=sec/3600;done+=1
    if (k+1)%20000==0:
        json.dump({'tag':tag,'th':TH_UP,'partial':True,'files':done,'hours':hours,'fa':fa},
                  open(os.path.expanduser(f'~/wuwexp/results/fulltrained/stress_{tag}.json'),'w'))
        print(f'[{tag}] {k+1} files {hours:.0f}h fa={fa} {(time.time()-t0)/60:.0f}min',flush=True)
json.dump({'tag':tag,'th':TH_UP,'partial':False,'files':done,'hours':round(hours,1),'fa':fa,'fa_per_hr':round(fa/hours,5)},
          open(os.path.expanduser(f'~/wuwexp/results/fulltrained/stress_{tag}.json'),'w'))
print(f'[{tag}] DONE fa={fa} hours={hours:.1f}',flush=True)
