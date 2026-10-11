#!/usr/bin/env python3
"""Coordinate the existing two Pods through protected Jupyter connections.

Scientific work runs only through committed repository scripts. The supplied
connection adapters execute short remote commands; credentials never enter logs.
No Pod is created, deleted, or stopped, and this coordinator does not train.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.parse import quote

import requests

BASE = '/workspace'
DATA = BASE+'/data/vllm_sft_preparation'
ARTIFACTS = BASE+'/artifacts/vllm_sft_preparation'
GENERATION_REPO = BASE+'/inducing_persistent_textual_chain_of_thought_illegibility'
PREPARATION_REPO = BASE+'/vllm_preparation_repo'
DEADLINE = '2026-10-11T20:20:00Z'
MODELS = ('7b','14b')


class Pod:
    def __init__(self, credential_file, adapter, controller_python):
        auth = json.loads(Path(credential_file).read_text())
        self.base, self.adapter, self.python = auth['base_url'].rstrip('/'), adapter, controller_python
        self.session = requests.Session()
        self.session.headers['Authorization'] = 'token '+auth['token']

    def read(self, path):
        response = self.session.get(self.base+'/api/contents/'+quote(path.lstrip('/')),
                                    params={'content':1,'format':'base64'},timeout=30)
        response.raise_for_status()
        return base64.b64decode(response.json()['content'])

    def put(self, path, content):
        response = self.session.put(self.base+'/api/contents/'+quote(path.lstrip('/')),
                        json={'type':'file','format':'base64','content':base64.b64encode(content).decode()},timeout=60)
        response.raise_for_status()

    def execute(self, code):
        with tempfile.NamedTemporaryFile(mode='w',dir='/workspace/scratch',suffix='.py') as f:
            f.write(code);f.flush()
            result = subprocess.run([self.python,self.adapter,f.name,'--timeout','45'],
                                    capture_output=True,text=True,timeout=55)
        if result.returncode:
            # Do not propagate arbitrary connection errors or secret-bearing URLs.
            raise RuntimeError('Jupyter command failed; inspect protected controller log')
        return next(json.loads(line) for line in reversed(result.stdout.splitlines()) if line.startswith('{'))

    def download(self, path, target, expected_hash):
        response = self.session.get(self.base+'/files/'+quote(path.lstrip('/')),
                                    stream=True,timeout=(30,60))
        response.raise_for_status()
        digest = hashlib.sha256()
        with target.open('wb') as output:
            for chunk in response.iter_content(1024*1024):
                output.write(chunk);digest.update(chunk)
        if digest.hexdigest()!=expected_hash:
            raise ValueError('Transferred raw cache checksum differs')


def env_code():
    return "env=os.environ.copy();env.update(GOOGLE_OAUTH_TOKEN_FILE='/workspace/.secrets/google_token.json',GOOGLE_DRIVE_FOLDER_ID='176a8Tm6l4WhfakY_0nCOo4ICeT7KPSmx',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')"


def launch_preparation(pod, phase, round_number, candidate_path):
    state = BASE+'/bootstrap/vllm_sft_preparation_state.json'
    return pod.execute(f"""
import json,os,subprocess
from pathlib import Path
state=Path({state!r});old=json.loads(state.read_text());stat=Path('/proc')/str(old['pid'])/'stat'
if stat.exists() and stat.read_text().split(')')[1].split()[0]!='Z':raise RuntimeError('Prior CPU worker is still active')
{env_code()}
env['CUDA_VISIBLE_DEVICES']=''
args=['/workspace/illegibility_gpu_env/bin/python','scripts/prepare_vllm_sft_data.py','--phase',{phase!r},'--config','/workspace/artifacts/collection_vllm_dc520cb/configuration.json','--candidates',{candidate_path!r},'--full-candidates','/workspace/collection_prerequisites_9742/data/candidates/manifest.json','--data-root',{DATA!r},'--artifact-root',{ARTIFACTS!r},'--cache-root','/workspace/artifacts/model_cache','--round',str({round_number}),'--workers','16','--chunk-size','128']
log=Path('/workspace/bootstrap/vllm_sft_preparation_'+{phase!r}+'_'+str({round_number}).zfill(3)+'.log')
with log.open('a') as output:p=subprocess.Popen(args,cwd={PREPARATION_REPO!r},env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
state.write_text(json.dumps({{'pid':p.pid,'args':args,'repo':{PREPARATION_REPO!r},'log':str(log),'round':{round_number},'phase':{phase!r}}}))
print(json.dumps({{'pid':p.pid,'phase':{phase!r},'round':{round_number}}}))
""")


def upgrade_preparation(pod):
    return pod.execute(f"""
import subprocess,json
subprocess.run(['git','fetch','--quiet','/workspace/bootstrap/vllm_preparation_build.bundle','HEAD'],cwd={PREPARATION_REPO!r},check=True)
subprocess.run(['git','checkout','--quiet','--detach','FETCH_HEAD'],cwd={PREPARATION_REPO!r},check=True)
print(json.dumps({{'preparation_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd={PREPARATION_REPO!r},text=True).strip()}}))
""")


def wait_cpu(pod, emit):
    while True:
        value = pod.execute(f"""
import json
from pathlib import Path
s=json.loads(Path('/workspace/bootstrap/vllm_sft_preparation_state.json').read_text());p=Path('/proc')/str(s['pid'])/'stat'
live=p.exists() and p.read_text().split(')')[1].split()[0]!='Z'
path=Path({ARTIFACTS!r})/'preparation_progress.json'
progress=json.loads(path.read_text()) if path.exists() else {{'status':'STARTING'}}
events=[]
for line in Path(s['log']).read_text().splitlines():
 if line.startswith('{{'):
  try:events.append(json.loads(line))
  except ValueError:pass
print(json.dumps({{'live':live,'progress':progress,'last_event':events[-1] if events else None}}))
""")
        emit(value['progress'])
        if not value['live']:
            if not value['last_event'] or value['last_event'].get('status') not in ('REFILL_REQUIRED','ACCEPTED_POOL_VERIFIED','COMPLETE'):
                raise RuntimeError('CPU preparation ended without a verified stage boundary')
            return value['last_event']
        time.sleep(20)


def launch_refill(pod, model, round_number, candidate_path):
    root = BASE+'/data/collection_vllm_refill_'+str(round_number).zfill(3)
    artifacts = BASE+'/artifacts/collection_vllm_refill_'+str(round_number).zfill(3)
    python = '/workspace/vllm_generation_env/bin/python' if model=='7b' else '/workspace/vllm_benchmark_env/bin/python'
    config = '/workspace/bootstrap/runtime_75eec060b1fe_batched.yaml' if model=='7b' else '/workspace/bootstrap/consumer_75eec060b1fe.yaml'
    value = pod.execute(f"""
import json,os,subprocess
from pathlib import Path
repo=Path({GENERATION_REPO!r})
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!='dc520cb73bf43b59a9566404dd4a7aa8358f1cc3':raise RuntimeError('Generation source changed')
state=Path('/workspace/bootstrap/refill_'+str({round_number}).zfill(3)+'_'+{model!r}+'_state.json')
if state.exists():
 previous=json.loads(state.read_text());stat=Path('/proc')/str(previous['pid'])/'stat'
 live=stat.exists() and stat.read_text().split(')')[1].split()[0]!='Z'
 complete='"event": "COMPLETE"' in Path(previous['log']).read_text()
 if not live and not complete:raise RuntimeError('Prior refill failed; inspect before replay')
 print(json.dumps({{'model':{model!r},'round':{round_number},'pid':previous['pid'],'reused_existing_worker':True}}))
{env_code()}
env.update(VLLM_WORKER_MULTIPROC_METHOD='spawn',CLEAN_GENERATION_BATCH_SIZE='128')
args=[{python!r},'scripts/run_vllm_generation_worker.py','--config',{config!r},'--candidates',{candidate_path!r},'--model',{model!r},'--data-root',{root!r},'--artifact-root',{artifacts!r},'--cache-root','/workspace/artifacts/model_cache','--deadline',{DEADLINE!r},'--run-id','vllm_fresh_dc520cb_refill_'+str({round_number}).zfill(3),'--batch-size','128']
log=Path('/workspace/bootstrap/refill_'+str({round_number}).zfill(3)+'_'+{model!r}+'.log')
if not state.exists():
 with log.open('w') as output:p=subprocess.Popen(args,cwd=repo,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
 state.write_text(json.dumps({{'pid':p.pid,'args':args,'log':str(log)}}))
 print(json.dumps({{'model':{model!r},'round':{round_number},'pid':p.pid}}))
""")
    return artifacts, root


def wait_refills(pods, round_number, roots, emit):
    while True:
        statuses = []
        for pod,model,root in zip(pods,MODELS,roots):
            value = pod.execute(f"""
from pathlib import Path
import json
progress=Path({root!r})/'progress.json';state=json.loads(progress.read_text()) if progress.exists() else {{'model':{model!r},'completed':0}}
events=[]
for line in Path('/workspace/bootstrap/refill_'+str({round_number}).zfill(3)+'_'+{model!r}+'.log').read_text().splitlines():
 if line.startswith('{{'):
  try:events.append(json.loads(line))
  except ValueError:pass
terminal=next((x for x in reversed(events) if x.get('event') in ('COMPLETE','PAUSED','FAILED')),None)
print(json.dumps({{'model':{model!r},'completed':state['completed'],'terminal':terminal}}))
""")
            if value['terminal'] and value['terminal']['event']!='COMPLETE':
                raise RuntimeError('Refill paused or failed; preserve verified data and diagnose')
            statuses.append(value)
        emit({'status':'REFILL_GENERATING','round':round_number,'models':statuses})
        if all(x['terminal'] and x['terminal']['event']=='COMPLETE' for x in statuses):
            return
        time.sleep(20)


def import_refills(pods, round_number, roots, local_root):
    for pod,model,root in zip(pods,MODELS,roots):
        info = pod.execute(f"""
from pathlib import Path
import json,hashlib,zipfile
source=Path({root!r})/'clean_traces/raw/production'/{model!r}
target=Path('/workspace/bootstrap/refill_transfer_'+str({round_number}).zfill(3)+'_'+{model!r}+'.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for path in source.glob('*.json'):z.write(path,path.name)
print(json.dumps({{'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}}))
""")
        if model=='14b':
            local=local_root/f'refill_{round_number:03d}_{model}.zip'
            pod.download(info['path'],local,info['sha256'])
            with local.open('rb') as f:
                count=0
                while chunk:=f.read(4*1024*1024):
                    pods[0].put(f'/workspace/bootstrap/refill_transfer_part_{count}.bin',chunk);count+=1
            pods[0].execute(f"""
from pathlib import Path
import hashlib,json
target=Path({info['path']!r})
with target.open('wb') as f:
 for i in range({count}):f.write(Path('/workspace/bootstrap/refill_transfer_part_'+str(i)+'.bin').read_bytes())
assert hashlib.sha256(target.read_bytes()).hexdigest()=={info['sha256']!r}
print(json.dumps({{'transfer':'VERIFIED'}}))
""")
        pods[0].execute(f"""
from pathlib import Path
import zipfile,hashlib,re,json
target=Path({DATA!r})/'clean_traces/raw/production'/{model!r};target.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile({info['path']!r}) as z:
 for name in z.namelist():
  if not re.fullmatch('[0-9a-f]{{64}}.json',name):raise RuntimeError('Unexpected cache member')
  output=target/name;content=z.read(name)
  if output.exists() and output.read_bytes()!=content:raise RuntimeError('Existing cache differs')
  output.write_bytes(content)
print(json.dumps({{'model':{model!r},'imported':len(z.namelist())}}))
""")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller-python',default='python')
    p.add_argument('--output-root',default='/workspace/scratch/sft_coordinator')
    a=p.parse_args()
    root=Path(a.output_root);root.mkdir(parents=True,exist_ok=True)
    pods=[Pod('/workspace/.secrets/runpod_jupyter'+suffix+'.json',
              '/workspace/scratch/runpod_control'+suffix+'.py',a.controller_python)
          for suffix in ('','_14b')]
    previous=None
    def emit(value):
        nonlocal previous
        if value!=previous:
            record={**value,'observed_at':datetime.now(timezone.utc).isoformat()}
            (root/'latest.json').write_text(json.dumps(record))
            with (root/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);previous=value
    # Source installation for the build phase occurs only after the currently
    # executing acceptance worker reaches a complete, verified boundary.
    round_number=0
    while True:
        result=wait_cpu(pods[0],emit)
        if result['status']=='COMPLETE':
            return
        if result['status']=='ACCEPTED_POOL_VERIFIED':
            upgrade_preparation(pods[0])
            launch_preparation(pods[0],'build',round_number,'/workspace/bootstrap/vllm_candidates.json')
            continue
        if datetime.now(timezone.utc)>=datetime.fromisoformat(DEADLINE.replace('Z','+00:00')):
            raise RuntimeError('Authorized worker deadline reached')
        round_number=result['round']+1
        candidate_path=result['refill_candidates']
        payload=pods[0].read(candidate_path)
        selected=json.loads(payload)
        if len(selected['candidates'])!=result['deficit']:
            raise ValueError('Refill size differs from the accepted deficit')
        candidate_path=f'/workspace/bootstrap/refill_candidates_{round_number:03d}.json'
        for pod in pods:
            pod.put(candidate_path,payload)
        launched=[launch_refill(pod,model,round_number,candidate_path) for pod,model in zip(pods,MODELS)]
        wait_refills(pods,round_number,[x[0] for x in launched],emit)
        import_refills(pods,round_number,[x[1] for x in launched],root)
        upgrade_preparation(pods[0])
        launch_preparation(pods[0],'accept',round_number,candidate_path)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'FAILED','error_type':type(error).__name__}),flush=True)
        raise SystemExit(1)
