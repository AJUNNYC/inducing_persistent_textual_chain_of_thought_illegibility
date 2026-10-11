"""Validate configurable Pod assignments without reading credential values."""
import json
from pathlib import Path
import re

DEFAULT_TOPOLOGY = Path(__file__).resolve().parents[1]/'deployment/coordinator.json'


def load_topology(path=DEFAULT_TOPOLOGY):
    value = json.loads(Path(path).read_text())
    pods = value['pods']
    if not pods or len({p['id'] for p in pods}) != len(pods):
        raise ValueError('Pod IDs must be nonempty and unique')
    ids = {p['id'] for p in pods}
    for pod in pods:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',pod['id']):
            raise ValueError('Invalid Pod ID')
        for field in ('credential_file','adapter'):
            if not Path(pod[field]).is_absolute():
                raise ValueError('Connection file paths must be absolute')
    if value['primary_pod'] not in ids:
        raise ValueError('Unknown primary Pod')
    models = set()
    assigned = set()
    for job in value['generation_workers']:
        if job['pod'] not in ids or job['model'] in models or job['pod'] in assigned:
            raise ValueError('Generation workers need unique models and dedicated known Pods')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',job['model']):
            raise ValueError('Invalid model assignment')
        for field in ('python','config'):
            if not Path(job[field]).is_absolute():
                raise ValueError('Generation runtime paths must be absolute')
        models.add(job['model']);assigned.add(job['pod'])
    if not models:
        raise ValueError('At least one generation worker is required')
    ranges = {}
    shard_ids = set()
    for shard in value.get('verification_shards',[]):
        if shard['pod'] not in ids or shard['pod']==value['primary_pod']:
            raise ValueError('Verification shard must use a known secondary Pod')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',shard['id']) or shard['id'] in shard_ids:
            raise ValueError('Verification shard IDs must be unique')
        shard_ids.add(shard['id'])
        start,stop,chunk=shard['start'],shard['stop'],shard['chunk_size']
        if not all(type(x) is int for x in (start,stop,chunk)) or not 0<=start<stop or not 1<=chunk<=128 or start%chunk:
            raise ValueError('Invalid or unaligned verification range')
        round_number=shard.get('round',0)
        if type(round_number) is not int or round_number<0:
            raise ValueError('Invalid verification round')
        for lo,hi in ranges.setdefault(round_number,[]):
            if start<hi and lo<stop:
                raise ValueError('Verification shard ranges overlap')
        ranges[round_number].append((start,stop))
        for field in ('data_root','state_file'):
            if not Path(shard[field]).is_absolute():
                raise ValueError('Verification runtime paths must be absolute')
    return value
