from pathlib import Path
import json,subprocess,time,hashlib
r=Path('/opt/runpod_work/tla6hi5owr4prx');root=r/'training_oct28'
for level in [1,3,7]:
 d=Path(f'/workspace/data/vllm_sft_preparation/transformed/kazitext/14b/level_{level:02d}')
 for name,count in [('train.jsonl',10000),('validation.jsonl',100)]:
  p=d/name;assert p.is_file(),str(p)
  with p.open() as f: actual=sum(1 for line in f if line.strip())
  assert actual==count,(str(p),actual)
 print('DATA_READY',level)
source=(r/'kazi_queue.py').read_text()
source=source.replace(" state('RUNNING',name,command=cmd)", " import shutil\n if shutil.disk_usage('/opt').free < 4*1024**3:raise RuntimeError('Less than 4 GiB local workspace free; backup queue stopped before starting job')\n state('RUNNING',name,command=cmd)")
prefix,body=source.split('try:\n for level in [5,10]:',1)
# The existing lock prevents overlapping training with the original coordinator.
prefix=prefix.replace('fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)', 'fcntl.flock(lock,fcntl.LOCK_EX)')
body=body.replace(" state('COMPLETE','all_requested_training',evaluation='pending')", " state('COMPLETE','kazitext_backup_levels_1_3_7',evaluation='pending')")
wait='''
original_pid=int((root.parent/'queue.pid').read_text())
while alive(original_pid):time.sleep(10)
previous=json.loads((root/'queue_status.json').read_text())
if previous['phase']!='COMPLETE':raise RuntimeError('Original KaziText queue did not complete; backup queue will not start')
(root.parent/'queue.pid').write_text(str(os.getpid()))
'''
# Hold a distinct coordinator lock while waiting; acquire training lock only after original queue exits.
prefix=prefix.replace("lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)","backup_lock=(root/'backup_queue.lock').open('a');fcntl.flock(backup_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)")
wait+="lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
code=prefix+wait+'try:\n for level in [1,3,7]:'+body
compile(code,'kazi_backup_queue.py','exec')
p=r/'kazi_backup_queue.py';p.write_text(code)
pidfile=r/'backup_queue.pid'
if pidfile.exists() and Path('/proc/'+pidfile.read_text().strip()).exists():raise RuntimeError('Backup queue already active')
with (r/'backup_queue.log').open('ab') as log:proc=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(p)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(proc.pid))
receipt={'order':[5,10,1,3,7],'epochs':[1,2],'backup_queue_pid':proc.pid,'original_queue_pid':int((r/'queue.pid').read_text()),'created_unix':time.time(),'script_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':'waiting_for_original_queue_success'}
(r/'backup_queue_plan.json').write_text(json.dumps(receipt,indent=2))
dest=Path('/workspace/pod_exports/tla6hi5owr4prx');(dest/'kazi_backup_queue.py').write_text(code);(dest/'kazi_backup_queue_plan.json').write_text(json.dumps(receipt,indent=2))
print('BACKUP_QUEUE_INSTALLED',json.dumps(receipt))
