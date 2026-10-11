#!/usr/bin/env python3
"""Import Drive-verified CPU acceptance chunks from configurable verification Pods.

The primary acceptance job retains candidate ordering, pool assembly and refills.
Only completed pair records and official transform caches are imported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from coordinate_vllm_sft_preparation import Pod, DATA
from coordinator_topology import DEFAULT_TOPOLOGY, load_topology


def run_shard(source, primary, shard, output_root):
    local = Path(output_root)/shard['id']
    local.mkdir(parents=True,exist_ok=True)
    begin,end,chunk_size = shard['start'],shard['stop'],shard['chunk_size']
    round_number = shard.get('round',0)
    for start in range(begin,end,chunk_size):
        ready = shard['data_root']+f'/verification_shards/round_{round_number:03d}/ready_{start:05d}.json'
        while True:
            result = source.execute(f"""
import json
from pathlib import Path
path=Path({ready!r})
state=json.loads(Path({shard['state_file']!r}).read_text())
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
        if info['start']!=start or info['stop']!=min(start+chunk_size,end):
            raise ValueError('Verification shard range differs')
        archive=local/f'transfer_{start:05d}.zip'
        source.download(info['archive'],archive,info['sha256'])
        with archive.open('rb') as f:
            parts=0
            while chunk:=f.read(4*1024*1024):
                primary.put(f'/workspace/bootstrap/verification_{shard["id"]}_part_{parts}.bin',chunk)
                parts+=1
        result=primary.execute(f"""
import hashlib,io,json,os,zipfile
from pathlib import Path
payload=b''.join(Path('/workspace/bootstrap/verification_'+{shard['id']!r}+'_part_'+str(i)+'.bin').read_bytes() for i in range({parts}))
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
print(json.dumps({{'status':'IMPORTED','shard':{shard['id']!r},'start':{start},'stop':{info['stop']},'files':len(names)}}))
""")
        with (local/'history.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True)
    print(json.dumps({'status':'SHARD_IMPORTED','shard':shard['id'],'stop':end}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology',default=str(DEFAULT_TOPOLOGY))
    parser.add_argument('--controller-python',default='python')
    parser.add_argument('--output-root',default='/workspace/scratch/verification_shard')
    args=parser.parse_args()
    topology=load_topology(args.topology)
    connections={entry['id']:Pod(entry['credential_file'],entry['adapter'],args.controller_python)
                 for entry in topology['pods']}
    primary=connections[topology['primary_pod']]
    shards=topology.get('verification_shards',[])
    if not shards:
        print(json.dumps({'status':'NO_SHARDS'}),flush=True)
        return
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        futures=[pool.submit(run_shard,connections[s['pod']],primary,s,args.output_root)
                 for s in shards]
        for future in futures:
            future.result()


if __name__=='__main__':
    main()
