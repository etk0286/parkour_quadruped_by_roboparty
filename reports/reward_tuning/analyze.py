from pathlib import Path
import struct, collections, csv, json
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
run=root/'roboparty_train/logs/rsl_rl/bigreddog_parkour_amp/2026-10-01_17-20-28_amp_flat'
out=Path(__file__).parent
values=collections.defaultdict(dict)
for p in sorted(run.glob('*tfevents*')):
 with p.open('rb') as f:
  while True:
   h=f.read(12)
   if len(h)<12: break
   n=struct.unpack('<Q',h[:8])[0]; b=f.read(n); crc=f.read(4)
   if len(b)!=n or len(crc)!=4: break
   e=Event.FromString(b)
   for v in e.summary.value:
    if v.HasField('simple_value'): values[v.tag][e.step]=v.simple_value
rows=[]
for tag,vs in sorted(values.items()):
 s=sorted(vs); a=np.array([vs[k] for k in s]); w=min(1000,len(a))
 rows.append(dict(tag=tag,count=len(a),first_step=s[0],last_step=s[-1],first_window_mean=float(a[:w].mean()),last_window_mean=float(a[-w:].mean()),last=float(a[-1])))
with (out/'metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(out/'summary.json').write_text(json.dumps({'run':str(run),'window':1000,'metrics':rows},indent=2))
groups=[('Episode_Termination/','Termination statistics'),('Metrics/','Task metrics'),('Curriculum/','Curriculum'),('Train/','Training')]
fig,axs=plt.subplots(4,1,figsize=(13,17))
for ax,(prefix,title) in zip(axs,groups):
 for tag,vs in sorted(values.items()):
  if tag.startswith(prefix):
   s=sorted(vs); block=max(1,len(s)//250)
   x=[np.mean(s[i:i+block]) for i in range(0,len(s),block)]
   y=[np.mean([vs[k] for k in s[i:i+block]]) for i in range(0,len(s),block)]
   ax.plot(x,y,label=tag[len(prefix):])
 ax.set_title(title+' (block means; mixed units)');ax.set_xlabel('Training iteration');ax.grid(alpha=.2)
 if ax.lines: ax.legend(fontsize=7,loc='best')
fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150)
r=[x for x in rows if x['tag'].startswith('Episode_Reward/')];r.sort(key=lambda x:abs(x['last_window_mean']))
fig,ax=plt.subplots(figsize=(11,9));ax.barh([x['tag'].split('/')[-1] for x in r],[x['last_window_mean'] for x in r]);ax.set_xlabel('Logged weighted reward statistic: last 1000 scalar samples');fig.tight_layout();fig.savefig(out/'reward_contributions.png',dpi=150)
print(json.dumps(rows,indent=1))
