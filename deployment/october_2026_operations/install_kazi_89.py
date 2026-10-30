from pathlib import Path
import json,tarfile,hashlib,subprocess
r=Path('/opt/runpod_work/tla6hi5owr4prx')
with tarfile.open('/workspace/bootstrap_oct28/kazi_extra_89.tar.gz') as t:
 for m in t.getmembers():
  assert m.name.startswith(('data/vllm_sft_preparation/transformed/kazitext/14b/level_08/','data/vllm_sft_preparation/transformed/kazitext/14b/level_09/'))
  dest=Path('/workspace')/m.name
  if dest.exists():
   assert hashlib.sha256(dest.read_bytes()).hexdigest()==hashlib.sha256(t.extractfile(m).read()).hexdigest()
  else:t.extract(m,'/workspace',filter='data')
for level in [8,9]:
 d=Path(f'/workspace/data/vllm_sft_preparation/transformed/kazitext/14b/level_{level:02d}')
 for split,count in [('train',10000),('validation',100)]:
  with (d/(split+'.jsonl')).open() as f:assert sum(bool(s.strip()) for s in f)==count
 print('DATA_READY',level,flush=True)
code=(r/'resume_kazi_queue.py').read_text().replace('for level in [1,3,7]:','for level in [8,9]:').replace('kazitext_backup_levels_1_3_7','kazitext_levels_8_9')
compile(code,'kazi_queue_89.py','exec');(r/'kazi_queue_89.py').write_text(code)
launcher='''from pathlib import Path
import json,subprocess,time
r=Path('/opt/runpod_work/tla6hi5owr4prx')
old=(r/'queue.pid').read_text().strip()
while True:
 p=Path('/proc/'+old+'/stat')
 if not p.exists() or p.read_text().split(') ',1)[1][0]=='Z':break
 time.sleep(5)
assert json.loads((r/'training_oct28/queue_status.json').read_text())['phase']=='COMPLETE','Previous queue must finish successfully'
with (r/'queue89.log').open('ab') as log:proc=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(r/'kazi_queue_89.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'queue.pid').write_text(str(proc.pid));(r/'backup_queue.pid').write_text(str(proc.pid))
print('KAZITEXT_8_9_STARTED',proc.pid,flush=True)
'''
p=r/'launch_kazi_89.py';p.write_text(launcher)
with (r/'launch89.log').open('ab') as log:proc=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(p)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'launch89.pid').write_text(str(proc.pid))
d=Path('/workspace/pod_exports/tla6hi5owr4prx')
for name in ['kazi_queue_89.py','launch_kazi_89.py']:(d/name).write_bytes((r/name).read_bytes())
print('QUEUE_INSTALLED',proc.pid,flush=True)
