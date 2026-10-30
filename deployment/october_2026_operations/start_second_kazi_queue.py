from pathlib import Path
import os,subprocess,json
r=Path('/opt/runpod_work/tla6hi5owr4prx');b=Path('/workspace/bootstrap_oct28')
p=Path('/tmp/training-private');p.chmod(0o700);(p/'google_token.json').chmod(0o600)
q=(b/'ordered_training_queue.py').read_text()
q=q.replace("root=Path('/workspace/training_oct28')",f"root=Path('{r}/training_oct28')")
q=q.replace("str(r/'launch/configuration_local_cache.json')",f"'{r}/kazi_configuration.json'")
q=q.replace('/workspace/artifacts/sft_14b_campaign_9de8813',str(r/'artifacts/sft_14b_campaign_9de8813'))
q=q.replace("TOKENIZERS_PARALLELISM='false'","TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1'")
start=q.index(' pid=int(');end=q.index(' for level in [5,10]:',start);q=q[:start]+q[end:];compile(q,'secondqueue','exec');(r/'kazi_queue.py').write_text(q)
script=r/'launch_when_ready.py'
script.write_text('''from pathlib import Path
import subprocess,time,os,json,shutil
r=Path('/opt/runpod_work/tla6hi5owr4prx');b=Path('/workspace/bootstrap_oct28')
while True:
 log=(r/'setup.log').read_text()
 if 'Installation and requested setup operations completed' in log:break
 if 'Traceback (most recent call last)' in log:raise RuntimeError('Runtime setup failed; inspect setup.log')
 time.sleep(10)
source=Path('/workspace/artifacts/sft_14b_campaign_9de8813');dest=r/'artifacts/sft_14b_campaign_9de8813'
if dest.exists():raise RuntimeError('Existing working state preserved; inspect before restarting launcher')
shutil.copytree(source,dest)
cfg=json.loads((b/'launch/configuration_local_cache.json').read_text());cfg['runtime']['artifact_root']=str(dest);cfg['runtime']['cache_root']='/opt/sft_model_cache';(r/'kazi_configuration.json').write_text(json.dumps(cfg))
e=os.environ.copy();e['PYTHONDONTWRITEBYTECODE']='1'
export=['/opt/illegibility_sft/bin/python','-u',str(r/'volume_checkpoint_exporter.py'),'--root',str(r),'--destination','/workspace/pod_exports/tla6hi5owr4prx','--staging','/opt/export_staging_tla']
f=(r/'exporter.log').open('a');p=subprocess.Popen(export,stdout=f,stderr=subprocess.STDOUT,env=e,start_new_session=True);(r/'exporter.pid').write_text(str(p.pid))
f=(r/'queue.log').open('a');p=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(r/'kazi_queue.py')],stdout=f,stderr=subprocess.STDOUT,env=e,start_new_session=True);(r/'queue.pid').write_text(str(p.pid));print('KAZITEXT_QUEUE_STARTED',p.pid,flush=True)
''')
f=(r/'launcher.log').open('a');p=subprocess.Popen(['python','-u',str(script)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(r/'launcher.pid').write_text(str(p.pid));print('KAZITEXT_LAUNCHER_STARTED',p.pid)
print('SETUP_TAIL',(r/'setup.log').read_text()[-700:])
