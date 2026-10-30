"""Publish immutable verified checkpoint archives; no shared locks or databases."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,tarfile,tempfile,time,uuid,fcntl

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def export(checkpoint,destination,staging):
 manifest=json.loads((checkpoint/'manifest.json').read_text())
 ident=manifest['checkpoint_hash'];fit=checkpoint.parent
 final=(fit/'fit_result.json').exists()
 key=hashlib.sha256((str(checkpoint)+ident+str(final)).encode()).hexdigest()
 marker=staging/(key+'.done')
 if marker.exists():return
 with tempfile.TemporaryDirectory(dir=staging) as temp:
  temp=Path(temp);payload=temp/'payload';payload.mkdir()
  target=payload/checkpoint.name;target.mkdir()
  for name,meta in manifest['files'].items():
   relative=Path(name)
   if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe checkpoint path')
   dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(checkpoint/relative,dest)
   if digest(dest)!=meta['sha256']:raise ValueError('Checkpoint copy hash mismatch')
  (target/'manifest.json').write_text(json.dumps(manifest))
  for name in ['run_identity.json','fit_status.json','runtime_evidence.json','fit_result.json','completion_receipt.json']:
   src=fit/name
   if src.exists():
    raw=src.read_bytes();json.loads(raw);(payload/name).write_bytes(raw)
  if final and (fit/'final_adapter').is_dir():shutil.copytree(fit/'final_adapter',payload/'final_adapter')
  archive=temp/'checkpoint.tar'
  with tarfile.open(archive,'w') as t:
   for p in sorted(payload.iterdir()):t.add(p,arcname=p.name)
  sha=digest(archive)
  destdir=destination/fit.name;destdir.mkdir(parents=True,exist_ok=True)
  dest=destdir/(checkpoint.name+'-'+uuid.uuid4().hex+'.tar')
  with archive.open('rb') as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst,8*1024*1024);dst.flush();os.fsync(dst.fileno())
  if digest(dest)!=sha:raise ValueError('Global volume readback mismatch')
  receipt={'status':'VERIFIED','archive':dest.name,'sha256':sha,'bytes':archive.stat().st_size,'checkpoint_hash':ident,'progress':manifest['progress'],'source_fit':str(fit),'final_result_present':final,'created_unix':time.time()}
  # Unique publication receipt written only after the archive verifies. Readers validate JSON and archive hash.
  with dest.with_suffix('.verified.json').open('x') as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
  marker.write_text(json.dumps(receipt));print(json.dumps(receipt),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--destination',required=True);p.add_argument('--staging',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
 staging=Path(a.staging);staging.mkdir(parents=True,exist_ok=True)
 while True:
  lock=(staging/"export.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX)
  errors=[]
  for manifest in sorted(Path(a.root).rglob('checkpoint_*/manifest.json')):
   try:export(manifest.parent,Path(a.destination),staging)
   except FileNotFoundError:pass # A retained older checkpoint may be pruned by its single writer.
   except Exception as e:errors.append({'checkpoint':str(manifest.parent),'error':type(e).__name__});print(json.dumps(errors[-1]),flush=True)
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  if a.once:
   if errors:raise RuntimeError('Export verification failed')
   return
  time.sleep(30)
if __name__=='__main__':main()
