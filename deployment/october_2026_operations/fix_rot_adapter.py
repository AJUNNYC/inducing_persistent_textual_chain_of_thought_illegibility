from pathlib import Path
import json,hashlib,shutil,subprocess,os,time
base=Path('/workspace/bootstrap_oct28');parents=base/'parents';fixed=[]
for level in [8,9]:
 p=parents/f'rot13_{level:02d}/final_adapter';m=json.loads((p/'manifest.json').read_text())
 for name,h in m['files'].items():
  dest=p/name
  if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest()==h:continue
  other=parents/f'rot13_{(9 if level==8 else 8):02d}/final_adapter'/name
  assert other.is_file() and hashlib.sha256(other.read_bytes()).hexdigest()==h,('No identical verified source',level,name)
  assert not dest.exists(),('Preserve existing differing file',str(dest))
  shutil.copyfile(other,dest)
  assert hashlib.sha256(dest.read_bytes()).hexdigest()==h
  fixed.append(str(dest))
print('RESTORED',fixed,flush=True)
script="""import sys,json
from pathlib import Path
sys.path.insert(0,'/workspace/bootstrap_oct28/source_71fe856/src')
from training.sft.individual_fit.sft_fit import verify_fit_result
for level in [8,9]:
 p=Path(f'/workspace/bootstrap_oct28/parents/rot13_{level:02d}')
 verify_fit_result(p,json.loads((p/'fit_result.json').read_text())['identity'])
 print('PARENT_FULL_VERIFICATION_PASSED',level,flush=True)
"""
res=subprocess.run(['/opt/illegibility_sft/bin/python','-c',script],capture_output=True,text=True)
print(res.stdout,flush=True)
if res.returncode:print(res.stderr[-1500:]);raise RuntimeError('Parent verification failed')
r=Path('/opt/runpod_work/x7j03ag24vz0gk');q=r/'ordered_training_queue.py';code=q.read_text()
start=code.index('try:\n pid=');end=code.index(' for level in [8,9]:',start)
code=code[:start]+'try:\n'+code[end:]
compile(code,'resume_rot89.py','exec');p=r/'resume_rot89.py';p.write_text(code)
state=json.loads((r/'training_oct28/queue_status.json').read_text());assert state['phase']=='STOPPED'
with (r/'queue.log').open('ab') as log:proc=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(p)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'queue.pid').write_text(str(proc.pid))
receipt={'restored':fixed,'verified_levels':[8,9],'queue_pid':proc.pid,'time':time.time()}
(r/'adapter_repair.json').write_text(json.dumps(receipt));(Path('/workspace/pod_exports/x7j03ag24vz0gk')/'adapter_repair.json').write_text(json.dumps(receipt))
print('RESUMED_ROT13_8_THEN_9',proc.pid,flush=True)
