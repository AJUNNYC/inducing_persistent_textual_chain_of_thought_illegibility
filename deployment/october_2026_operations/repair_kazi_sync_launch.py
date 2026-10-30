from pathlib import Path
import subprocess
r=Path('/opt/runpod_work/tla6hi5owr4prx')
code='''from pathlib import Path
import sys,json,os,subprocess
sys.path.insert(0,'/workspace/bootstrap_oct28/source_89609d1/src')
from operations.artifacts import ArtifactService
from operations.drive import DriveStore,drive_service
from shared_procedures.recovery_procedure import configure_worker_recovery
r=Path('/opt/runpod_work/tla6hi5owr4prx');output=r/'training_oct28/kazitext_level03_epoch2'
configure_worker_recovery(output,worker_id='sft_kazitext_14b_level_03_continuation_v2')
service=ArtifactService(output,output/'drive_registry.json',DriveStore(drive_service('/tmp/training-private/google_token.json'),'176a8Tm6l4WhfakY_0nCOo4ICeT7KPSmx'),drive_recovery=True)
result=service.sync_artifacts();print('DRIVE_REPAIR',json.dumps(result),flush=True)
assert result['status']=='VERIFIED','Drive repair failed; do not launch'
service.export_manifest()
code=(r/'resume_kazi_queue.py').read_text().replace('for level in [1,3,7]:','for level in [3,7]:')
p=r/'resume_kazi_after_sync.py';p.write_text(code);compile(code,str(p),'exec')
assert json.loads((r/'training_oct28/queue_status.json').read_text())['phase']=='STOPPED'
with (r/'backup_queue.log').open('ab') as log:proc=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(p)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'queue.pid').write_text(str(proc.pid));(r/'backup_queue.pid').write_text(str(proc.pid))
print('QUEUE_RESUMED',proc.pid,flush=True)
'''
p=r/'repair_drive_sync.py';p.write_text(code)
with (r/'drive_sync_repair.log').open('ab') as log:proc=subprocess.Popen(['/opt/illegibility_sft/bin/python','-u',str(p)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('REPAIR_STARTED',proc.pid)
