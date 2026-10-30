"""Operational serial queue. All scientific work uses pinned repository commands."""
from pathlib import Path
import os,json,time,subprocess,signal,fcntl
r=Path('/workspace/bootstrap_oct28');new=r/'source_89609d1';old=r/'source_9de8813';py='/opt/illegibility_sft/bin/python'
root=Path('/workspace/training_oct28');root.mkdir(exist_ok=True)
lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
newsha=json.loads((r/'control_configuration.json').read_text())['runtime']['implementation_revision']
stop={'requested':False}
def pause(*_):stop['requested']=True
signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
def state(phase,job,**kw):
 d={'phase':phase,'job':job,'updated_unix':time.time(),**kw};t=root/'queue_status.tmp';t.write_text(json.dumps(d));t.replace(root/'queue_status.json');print(json.dumps(d),flush=True)
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
 except FileNotFoundError:return False
def env(sha):
 e=os.environ.copy();e.update(SOURCE_CODE_COMMIT=sha,GOOGLE_OAUTH_TOKEN_FILE='/tmp/training-private/google_token.json',GOOGLE_DRIVE_FOLDER_ID='176a8Tm6l4WhfakY_0nCOo4ICeT7KPSmx',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false');return e
def run(name,cmd,sha,cwd):
 state('RUNNING',name,command=cmd)
 with (root/(name+'.log')).open('ab') as log:
  p=subprocess.Popen(cmd,cwd=cwd,env=env(sha),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  sent=False
  while p.poll() is None:
   if (stop['requested'] or (root/'PAUSE_REQUESTED').exists()) and not sent:p.send_signal(signal.SIGTERM);sent=True
   time.sleep(3)
 if p.returncode!=0:raise RuntimeError(f'{name} exited {p.returncode}; queue stopped')
 if stop['requested']:raise RuntimeError('Queue paused')
def continuation(family,level,parent,trainroot):
 out=root/f'{family}_level{level:02d}_epoch2'
 return [py,'-u',str(new/'scripts/continue_sft_level.py'),'--family',family,'--level',str(level),'--additional-epochs','1','--learning-rate','5e-5','--parent-fit',str(parent),'--parent-checkpoint',str(parent/'checkpoint_000625'),'--train',str(trainroot/'train.jsonl'),'--validation',str(trainroot/'validation.jsonl'),'--output',str(out),'--model-cache','/opt/sft_model_cache']
try:
 pid=int((r/'rot13_control.pid').read_text());state('WAITING_FOR_CONTROL','rot13_level00',pid=pid)
 while alive(pid):
  if stop['requested'] or (root/'PAUSE_REQUESTED').exists():os.kill(pid,signal.SIGTERM);raise RuntimeError('Requested safe pause')
  time.sleep(3)
 result=json.loads((root/'rot13_level00/fit_result.json').read_text())
 assert result['status']=='complete' and result['progress']['completed_steps']==625
 assert '"drive_status": "VERIFIED"' in (r/'rot13_control.log').read_text()
 if not (r/'parents_restored.json').exists():raise RuntimeError('Parent restore incomplete; do not advance')
 for level in [8,9]:
  run(f'rot13_level{level:02d}_epoch2',continuation('rot13',level,r/f'parents/rot13_{level:02d}',r/f'rot13_data/level_{level:02d}'),newsha,new)
 for level in [5,10]:
  cmd=[py,'-u',str(old/'scripts/run_certified_sft.py'),'--config',str(r/'launch/configuration_local_cache.json'),'--certificate',str(r/'launch/handoff_certificate.json'),'--worklist',str(r/'launch/worklist.json'),'--preflight',str(r/'launch/preflight.json'),'--bundle','/workspace/data/vllm_sft_preparation/transformed/manifest.json','--family','kazitext','--model','14b','--level',str(level),'--deadline','2026-11-05T00:00:00Z']
  run(f'kazitext_level{level:02d}_epoch1',cmd,'9de8813031da737453f491723a10b305679dc42d',old)
  parent=Path(f'/workspace/artifacts/sft_14b_campaign_9de8813/training/sft/kazitext/14b/level_{level:02d}')
  run(f'kazitext_level{level:02d}_epoch2',continuation('kazitext',level,parent,Path(f'/workspace/data/vllm_sft_preparation/transformed/kazitext/14b/level_{level:02d}')),newsha,new)
 state('COMPLETE','all_requested_training',evaluation='pending')
except BaseException as exc:
 state('STOPPED','inspect_log',error_type=type(exc).__name__,detail=str(exc));raise
