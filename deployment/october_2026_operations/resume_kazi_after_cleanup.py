from pathlib import Path
import json,subprocess,time
r=Path('/opt/runpod_work/tla6hi5owr4prx');p=r/'kazi_backup_queue.py';code=p.read_text()
a=code.index('original_pid=int(');b=code.index("lock=(root/'queue.lock')",a)
code=code[:a]+"(root.parent/'queue.pid').write_text(str(os.getpid()))\n"+code[b:]
needle="  run(f'kazitext_level{level:02d}_epoch1',cmd,'9de8813031da737453f491723a10b305679dc42d',old)"
replacement="  existing=root.parent/f'artifacts/sft_14b_campaign_9de8813/training/sft/kazitext/14b/level_{level:02d}/fit_result.json'\n  if not (existing.exists() and json.loads(existing.read_text()).get('status')=='complete'):\n "+needle
assert needle in code;code=code.replace(needle,replacement)
needle=" if stop['requested']:raise RuntimeError('Queue paused')"
replacement=needle+"\n subprocess.run([py,str(root.parent/'volume_checkpoint_exporter.py'),'--root',str(root.parent),'--destination','/workspace/pod_exports/tla6hi5owr4prx','--staging','/opt/export_staging_tla','--once'],check=True)\n subprocess.run([py,str(root.parent/'verified_cache_cleanup.py')],check=True)"
code=code.replace(needle,replacement)
compile(code,'resume_kazi_queue.py','exec');(r/'resume_kazi_queue.py').write_text(code)
launcher='''from pathlib import Path
import subprocess,json,time
r=Path('/opt/runpod_work/tla6hi5owr4prx')
pid=(r/'cleanup.pid').read_text().strip()
while True:
 p=Path('/proc/'+pid+'/stat')
 if not p.exists() or p.read_text().split(') ',1)[1][0]=='Z':break
 time.sleep(5)
assert 'FREED_GB' in (r/'cleanup.log').read_text(),'Cleanup did not finish successfully'
assert json.loads((r/'training_oct28/queue_status.json').read_text())['phase']=='STOPPED'
with (r/'backup_queue.log').open('ab') as f:
 proc=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(r/'resume_kazi_queue.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(r/'backup_queue.pid').write_text(str(proc.pid));(r/'queue.pid').write_text(str(proc.pid))
print('RESUMED_KAZITEXT',proc.pid,flush=True)
'''
p=r/'resume_after_cleanup.py';p.write_text(launcher)
with (r/'cleanup_resume.log').open('ab') as f:proc=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(p)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('RESUME_WATCHER',proc.pid)
dest=Path('/workspace/pod_exports/tla6hi5owr4prx')
for name in ['resume_kazi_queue.py','verified_cache_cleanup.py','resume_after_cleanup.py']:(dest/name).write_bytes((r/name).read_bytes())
