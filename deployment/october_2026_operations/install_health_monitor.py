from pathlib import Path
import subprocess,json
r=next(Path('/opt/runpod_work').iterdir())
code='''from pathlib import Path
import json,time,subprocess,os
r=Path(__file__).parent
while True:
 try:
  report={'time':time.time(),'pod':r.name,'alerts':[],'fits':[]}
  report['queue']=json.loads((r/'training_oct28/queue_status.json').read_text())
  if report['queue']['phase']=='STOPPED':report['alerts'].append('Queue stopped')
  for name in ['queue','exporter']:
   p=r/(name+'.pid');pid=p.read_text().strip()
   alive=Path('/proc/'+pid).exists();report[name+'_alive']=alive
   if not alive and report['queue']['phase']!='COMPLETE':report['alerts'].append(name+' missing')
  for p in r.rglob('fit_status.json'):
   x=json.loads(p.read_text())
   if x.get('status')!='running':continue
   saves=sorted(p.parent.glob('checkpoint_*/manifest.json'))
   if saves:
    latest=saves[-1];m=json.loads(latest.read_text());age=time.time()-latest.stat().st_mtime
    report['fits'].append({'path':str(p.parent),'saved_steps':m['progress']['completed_steps'],'checkpoint_age_seconds':round(age)})
    if age>3600:report['alerts'].append('No checkpoint for over one hour at '+str(p.parent))
  report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
  space=os.statvfs('/opt');report['free_disk_gb']=round(space.f_bavail*space.f_frsize/1e9,2)
  if report['free_disk_gb']<3:report['alerts'].append('Local disk below 3 GB')
  tmp=r/'health_status.tmp';tmp.write_text(json.dumps(report));tmp.replace(r/'health_status.json')
  with (r/'health_history.jsonl').open('a') as f:f.write(json.dumps(report)+'\\n')
 except Exception as e:
  with (r/'health_errors.log').open('a') as f:f.write(type(e).__name__+': '+str(e)+'\\n')
 time.sleep(60)
'''
p=r/'health_monitor.py';p.write_text(code);compile(code,str(p),'exec')
pidfile=r/'health_monitor.pid'
if pidfile.exists() and Path('/proc/'+pidfile.read_text().strip()).exists():print('MONITOR_ALREADY_RUNNING')
else:
 with (r/'health_monitor.log').open('ab') as log:proc=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(p)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 pidfile.write_text(str(proc.pid));print('MONITOR_STARTED',r.name,proc.pid)
receipts=sorted((Path('/workspace/pod_exports')/r.name).rglob('*.verified.json'),key=lambda p:p.stat().st_mtime,reverse=True)
for p in receipts[:2]:
 x=json.loads(p.read_text());print('VOLUME_BACKUP',p.parent.name,x.get('progress',{}).get('completed_steps'),x.get('bytes'))
