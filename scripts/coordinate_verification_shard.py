#!/usr/bin/env python3
"""Import Drive-verified CPU acceptance chunks from the second existing Pod.

The primary acceptance job retains candidate ordering, pool assembly and refills.
Only completed pair records and official transform caches are imported.
"""
import hashlib
import json
from pathlib import Path
import time

from coordinate_vllm_sft_preparation import Pod, DATA


def main():
    pods = [Pod('/workspace/.secrets/runpod_jupyter'+suffix+'.json',
                '/workspace/scratch/runpod_control'+suffix+'.py','python')
            for suffix in ('','_14b')]
    local = Path('/workspace/scratch/verification_shard')
    local.mkdir(parents=True,exist_ok=True)
    for start in range(2048,10100,128):
        ready = DATA+f'/verification_shards/round_000/ready_{start:05d}.json'
        while True:
            result = pods[1].execute(f"""
import json
from pathlib import Path
path=Path({ready!r})
state=json.loads(Path('/workspace/bootstrap/verification_shard_state.json').read_text())
stat=Path('/proc')/str(state['pid'])/'stat'
alive=stat.exists() and stat.read_text().split(')')[1].split()[0]!='Z'
print(json.dumps({{'ready':json.loads(path.read_text()) if path.exists() else None,'alive':alive}}))
""")
            if result['ready']:
                break
            if not result['alive']:
                raise RuntimeError('Verification shard stopped before its next verified chunk')
            time.sleep(20)
        info=result['ready']
        if info['start']!=start or info['stop']!=min(start+128,10100):
            raise ValueError('Verification shard range differs')
        archive=local/f'transfer_{start:05d}.zip'
        pods[1].download(info['archive'],archive,info['sha256'])
        with archive.open('rb') as f:
            parts=0
            while chunk:=f.read(4*1024*1024):
                pods[0].put(f'/workspace/bootstrap/verification_part_{parts}.bin',chunk)
                parts+=1
        result=pods[0].execute(f"""
import hashlib,io,json,os,zipfile
from pathlib import Path
payload=b''.join(Path('/workspace/bootstrap/verification_part_'+str(i)+'.bin').read_bytes() for i in range({parts}))
if hashlib.sha256(payload).hexdigest()!={info['sha256']!r}:raise ValueError('Verification transfer checksum differs')
root=Path({DATA!r})
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    names=z.namelist()
    if len(names)!=len(set(names)):raise ValueError('Duplicate archive member')
    for name in names:
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in ('clean_traces','transform_cache') or rel.suffix!='.json':raise ValueError('Invalid verification cache member')
        path=root/rel;value=z.read(name)
        json.loads(value)
        if path.exists():
            if json.loads(path.read_bytes())!=json.loads(value):raise ValueError('Existing verification cache differs')
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.shard.tmp');temp.write_bytes(value);os.replace(temp,path)
print(json.dumps({{'status':'IMPORTED','start':{start},'stop':{info['stop']},'files':len(names)}}))
""")
        with (local/'history.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True)
    print(json.dumps({'status':'SHARD_IMPORTED','stop':10100}),flush=True)


if __name__=='__main__':
    main()
