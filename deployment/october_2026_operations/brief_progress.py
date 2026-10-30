from pathlib import Path
import json,subprocess,time
r=next(Path('/opt/runpod_work').iterdir());q=json.loads((r/'training_oct28/queue_status.json').read_text());print('POD',r.name,'JOB',q['job'],'PHASE',q['phase'])
for p in r.rglob('fit_status.json'):
 result=p.parent/'fit_result.json'
 if result.exists():
  x=json.loads(result.read_text());print('RESULT',p.parent.name,x.get('status'));continue
 x=json.loads(p.read_text());prog=x.get('progress',{});saves=sorted(p.parent.glob('checkpoint_*/manifest.json'))
 print('FIT',p.parent.name,x.get('status'),'LIVE_STEPS',prog.get('completed_steps'),'SAVED',json.loads(saves[-1].read_text()).get('progress',{}).get('completed_steps') if saves else None)
for p in (r/'training_oct28').glob('*.log'):
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-500));tail=f.read().decode(errors='replace')
 if 'Error' in tail or 'Traceback' in tail: print('ERROR_LOG',p.name,tail)
p=r/'health_status.json'
if p.exists():
 x=json.loads(p.read_text());print('HEALTH',x.get('alerts'),'FREE_GB',x.get('free_disk_gb'))
for name in ['exporter','backup_queue']:
 p=r/(name+'.pid')
 if p.exists():print(name,'ALIVE',Path('/proc/'+p.read_text().strip()).exists())
print(subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout)
receipts=sorted((Path('/workspace/pod_exports')/r.name).rglob('*.verified.json'),key=lambda p:p.stat().st_mtime,reverse=True)
for p in receipts[:2]:
 x=json.loads(p.read_text());print('VOLUME_BACKUP',p.parent.name,x.get('progress',{}).get('completed_steps'),x.get('final_result_present'))
