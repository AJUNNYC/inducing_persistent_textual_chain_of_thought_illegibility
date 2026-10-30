from pathlib import Path
import json
r=next(Path('/opt/runpod_work').iterdir());dest=Path('/workspace/pod_exports')/r.name
print('POD',r.name)
for d in sorted(dest.iterdir()):
 if not d.is_dir():continue
 rows=[]
 for p in d.glob('*.verified.json'):
  x=json.loads(p.read_text())
  if x.get('status')=='VERIFIED' and (d/x.get('archive','')).is_file():rows.append(x)
 if rows:
  x=max(rows,key=lambda x:(x.get('progress',{}).get('completed_steps',-1),x.get('final_result_present',False),x.get('created_unix',0)))
  print(d.name,'STEP',x.get('progress',{}).get('completed_steps'),'FINAL',x.get('final_result_present'),'PATH',str(d))
