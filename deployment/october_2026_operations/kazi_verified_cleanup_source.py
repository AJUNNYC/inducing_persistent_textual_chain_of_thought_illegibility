"""Evict only completed-fit local recovery copies with verified volume archives."""
from pathlib import Path
import json,hashlib,shutil,fcntl,time
ROOT=Path('/opt/runpod_work/tla6hi5owr4prx')
DEST=Path('/workspace/pod_exports/tla6hi5owr4prx')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 lock=Path('/opt/export_staging_tla/export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 removed=[];freed=0
 for result in ROOT.rglob('fit_result.json'):
  x=json.loads(result.read_text())
  if x.get('status')!='complete':continue
  fit=result.parent
  receipts=[]
  for p in (DEST/fit.name).glob('*.verified.json'):
   y=json.loads(p.read_text())
   if y.get('status')=='VERIFIED' and y.get('source_fit')==str(fit):receipts.append((p,y))
  for checkpoint in sorted(fit.glob('checkpoint_*')):
   if not checkpoint.is_dir() or checkpoint.name=='checkpoint_000625':continue
   m=json.loads((checkpoint/'manifest.json').read_text())
   matches=[(p,y) for p,y in receipts if y.get('checkpoint_hash')==m['checkpoint_hash']]
   if not matches:raise RuntimeError('No volume backup for '+str(checkpoint))
   p,y=matches[-1];archive=p.parent/y['archive']
   if archive.stat().st_size!=y['bytes'] or digest(archive)!=y['sha256']:raise RuntimeError('Archive verification failed '+str(archive))
   size=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file())
   shutil.rmtree(checkpoint);freed+=size;removed.append({'local_copy':str(checkpoint),'preserved_archive':str(archive),'sha256':y['sha256']})
   print('LOCAL_COPY_EVICTED',str(checkpoint),flush=True)
 receipt={'time':time.time(),'freed_bytes':freed,'copies':removed,'preserved':'all volume archives, final adapters, final full checkpoints and result metadata'}
 with (ROOT/'verified_cache_cleanup.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
 print('FREED_GB',round(freed/1e9,2),flush=True)
if __name__=='__main__':main()
