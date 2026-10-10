"""Byte-cache/network fixtures exercise real digests and persistent retry limits."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import hashlib
import json
import urllib.error

import pytest
from huggingface_hub import hf_hub_download

import operations.assets as assets
from shared_procedures.recovery_procedure import RecoveryManager, RecoveryStopped
from shared_procedures.storage import read_json, atomic_write_json

REVISION = 'a'*40
OTHER_REVISION = 'b'*40


class Response(BytesIO):
    def __init__(self, payload, *, status=200, headers=None):
        super().__init__(payload)
        self.status=status
        self.headers=headers or {}


class NoWaitRecovery(RecoveryManager):
    def run(self, operation_id, operation, **kwargs):
        return super().run(operation_id, operation, sleep=lambda _:None)


@pytest.fixture(autouse=True)
def bounded_recovery(monkeypatch,tmp_path):
    monkeypatch.setattr(assets,'RecoveryManager',NoWaitRecovery)
    monkeypatch.delenv('HF_TOKEN',raising=False)
    monkeypatch.setenv('ILLEGIBILITY_WORKER_RECOVERY_PATH',str(tmp_path/'state'/'asset_retries.json'))


def sibling(name,payload,*,lfs=False):
    result={'rfilename':name,'size':len(payload)}
    if lfs:
        result['lfs']={'sha256':hashlib.sha256(payload).hexdigest(),'size':len(payload)}
    else:
        result['blobId']=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
    return result


def install_upstream(monkeypatch,versions=None):
    versions=versions or {REVISION:{'config.json':b'{"model_type":"qwen2"}',
        'tokenizer.json':b'fixture-tokenizer','tokenizer_config.json':b'{}','model.safetensors':b'fixture-weights'}}
    calls=[]
    def open_url(request,timeout):
        calls.append((request.full_url,dict(request.header_items())))
        if '/api/' in request.full_url:
            revision=request.full_url.split('/revision/')[1].split('?')[0]
            payloads=versions[revision]
            info={'sha':revision,'siblings':[sibling(name,payload,lfs=name.endswith('.safetensors')) for name,payload in payloads.items()]}
            return Response(json.dumps(info).encode())
        revision=request.full_url.split('/raw/')[-1].split('/resolve/')[-1].split('/')[0]
        name=request.full_url.split('/'+revision+'/')[1]
        payload=versions[revision][name]
        offset=request.headers.get('Range')
        if offset:
            start=int(offset.split('=')[1].split('-')[0])
            return Response(payload[start:],status=206,headers={'Content-Range':f'bytes {start}-{len(payload)-1}/{len(payload)}'})
        return Response(payload)
    monkeypatch.setattr(assets,'_open_url',open_url)
    return calls,versions


def asset(*,revision=REVISION,tokenizer_revision=REVISION):
    return {'asset_id':'model_7b','repo_id':'fixture/model','repo_type':'model','revision':revision,
            'tokenizer_revision':tokenizer_revision,'destination':'models/7b','license_authorization_required':False}


def test_shared_hub_cache_reuses_tokenizers_and_weights_with_no_second_download(tmp_path,monkeypatch):
    calls,versions=install_upstream(monkeypatch)
    tokenizer=assets.provision_asset(asset(),tmp_path,tokenizer_only=True)
    assert 'model.safetensors' not in tokenizer['files']
    full=assets.provision_asset(asset(),tmp_path)
    before=len(calls)
    kwargs=assets.model_load_kwargs('fixture/model',REVISION,tmp_path,offline=True)
    assert kwargs == {'cache_dir':str(tmp_path/'hub'),'local_files_only':True}
    weight=Path(hf_hub_download('fixture/model','model.safetensors',revision=REVISION,**kwargs))
    public=tmp_path/'models'/'7b'/REVISION/'full'/'model.safetensors'
    assert weight.read_bytes() == versions[REVISION]['model.safetensors']
    assert weight.stat().st_ino == public.stat().st_ino
    assert full['manifest_version']==2 and len(calls)==before
    assert len([url for url,_ in calls if url.endswith('/tokenizer.json')])==1
    assert len([url for url,_ in calls if url.endswith('/model.safetensors')])==1


def test_offline_tokenizer_selection_from_verified_full_snapshot(tmp_path,monkeypatch):
    calls,_=install_upstream(monkeypatch)
    assets.provision_asset(asset(),tmp_path)
    before=len(calls)
    kwargs=assets.model_load_kwargs('fixture/model',REVISION,tmp_path,tokenizer_only=True,offline=True)
    assert kwargs['local_files_only'] and len(calls)==before


def test_separate_tokenizer_revision_is_verified_in_own_snapshot(tmp_path,monkeypatch):
    calls,_=install_upstream(monkeypatch,{REVISION:{'config.json':b'{}','model.safetensors':b'original-weights'},
        OTHER_REVISION:{'config.json':b'{}','tokenizer.json':b'different-tokenizer'}})
    manifest=assets.provision_asset(asset(tokenizer_revision=OTHER_REVISION),tmp_path,tokenizer_only=True)
    assert manifest['revision']==REVISION and manifest['selected_revision']==OTHER_REVISION
    kwargs=assets.model_load_kwargs('fixture/model',OTHER_REVISION,tmp_path,tokenizer_only=True,offline=True)
    assert Path(hf_hub_download('fixture/model','tokenizer.json',revision=OTHER_REVISION,**kwargs)).read_bytes()==b'different-tokenizer'
    assert not any('/'+REVISION+'/' in url for url,_ in calls)


def test_manifest_identity_and_shared_blob_corruption_are_refused(tmp_path,monkeypatch):
    install_upstream(monkeypatch)
    assets.provision_asset(asset(),tmp_path,tokenizer_only=True)
    path=tmp_path/'models'/'7b'/REVISION/'tokenizer'
    marker=path/'asset_manifest.json'
    manifest=read_json(marker)
    manifest['repo_id']='fixture/another-model'
    atomic_write_json(marker,manifest)
    with pytest.raises(RuntimeError,match='manifest hash'):
        assets.validate_asset(path)
    manifest['repo_id']='fixture/model'
    atomic_write_json(marker,manifest)
    (path/'tokenizer.json').write_bytes(b'corrupted')
    with pytest.raises(RuntimeError,match='checksum'):
        assets.model_load_kwargs('fixture/model',REVISION,tmp_path,tokenizer_only=True)


def test_upstream_digest_mismatch_never_publishes_complete_manifest(tmp_path,monkeypatch):
    calls,_=install_upstream(monkeypatch)
    original=assets._open_url
    def corrupted(request,timeout):
        if request.full_url.endswith('/tokenizer.json'):
            return Response(b'bad--tokenizer---')
        return original(request,timeout)
    monkeypatch.setattr(assets,'_open_url',corrupted)
    with pytest.raises(RuntimeError,match='digest|byte size|exceeds'):
        assets.provision_asset(asset(),tmp_path,tokenizer_only=True)
    assert not (tmp_path/'models'/'7b'/REVISION/'tokenizer'/'asset_manifest.json').exists()
    assert len([url for url,_ in calls if url.endswith('/tokenizer.json')])==0


def test_interrupted_transfer_resumes_exact_range_and_verifies_actual_blob(tmp_path,monkeypatch):
    payload=b'fixture-weights'
    calls=[]
    class Interrupted(Response):
        def read(self,size=-1):
            if self.tell()==0:return super().read(7)
            raise TimeoutError('fixture connection interruption')
    def open_url(request,timeout):
        calls.append(dict(request.header_items()))
        if len(calls)==1:return Interrupted(payload)
        assert request.headers['Range']=='bytes=7-'
        return Response(payload[7:],status=206,headers={'Content-Range':f'bytes 7-{len(payload)-1}/{len(payload)}'})
    monkeypatch.setattr(assets,'_open_url',open_url)
    expected=assets._upstream_file(sibling('model.safetensors',payload,lfs=True))
    manager=NoWaitRecovery(tmp_path/'retries.json')
    blob=tmp_path/'blobs'/expected['upstream_hash']
    result=assets._download_blob(asset(),REVISION,expected,blob,manager)
    assert result['sha256']==hashlib.sha256(payload).hexdigest() and blob.read_bytes()==payload
    assert len(calls)==2 and manager.counters()['worker_retries']==1
    assert not blob.with_name(blob.name+'.incomplete').exists()


def test_retry_bound_persists_and_does_not_restart_original_attempt(tmp_path,monkeypatch):
    calls=[]
    def unavailable(request,timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url,503,'fixture transient failure',{},None)
    monkeypatch.setattr(assets,'_open_url',unavailable)
    for _ in range(2):
        with pytest.raises(RecoveryStopped):
            assets.provision_asset(asset(),tmp_path,tokenizer_only=True)
    assert len(calls)==4 # Initial request and exactly three authorized retries.
    counters=RecoveryManager(tmp_path/'state'/'asset_retries.json').counters()
    assert counters['worker_retries']==3 and list(counters['operations'].values())==[3]


def test_immutable_missing_or_unauthorized_inputs_make_no_network_requests(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(assets,'_open_url',lambda *a,**k:calls.append(a))
    with pytest.raises(ValueError,match='immutable'):
        assets.model_load_kwargs('fixture/model','main',tmp_path)
    with pytest.raises(RuntimeError,match='frozen'):
        assets.provision_asset({**asset(),'revision':None},tmp_path,offline=True)
    with pytest.raises(RuntimeError,match='authorization'):
        assets.provision_asset({**asset(),'license_authorization_required':True},tmp_path)
    with pytest.raises(RuntimeError,match='offline'):
        assets.provision_asset(asset(),tmp_path,offline=True)
    assert calls==[]


def test_snapshot_corruption_is_refused_even_when_manifest_payloads_are_intact(tmp_path,monkeypatch):
    install_upstream(monkeypatch)
    assets.provision_asset(asset(),tmp_path,tokenizer_only=True)
    snapshot=tmp_path/'hub'/'models--fixture--model'/'snapshots'/REVISION/'tokenizer.json'
    snapshot.unlink()
    snapshot.write_bytes(b'wrong-snapshot')
    with pytest.raises(RuntimeError,match='byte size|digest'):
        assets.model_load_kwargs('fixture/model',REVISION,tmp_path,tokenizer_only=True,offline=True)


def test_redirect_strips_bearer_token_before_leaving_hub_origin():
    import urllib.request
    original=urllib.request.Request('https://huggingface.co/fixture/model',headers={'Authorization':'Bearer fixture-secret'})
    redirected=assets._SafeRedirect().redirect_request(original,None,302,'',{},'https://cdn-lfs.hf.co/signed')
    assert redirected.get_header('Authorization') is None
    with pytest.raises(RuntimeError,match='TLS'):
        assets._SafeRedirect().redirect_request(original,None,302,'',{},'http://cdn-lfs.hf.co/signed')


@pytest.mark.parametrize('name',['../escape','/absolute','folder/../escape','folder\\escape','folder//file','.git/config'])
def test_repository_paths_cannot_escape_cache(name):
    with pytest.raises(RuntimeError,match='Unsafe'):
        assets._relative_name(name)
