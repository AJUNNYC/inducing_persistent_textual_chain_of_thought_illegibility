import pytest
from shared_procedures.identity import content_hash
from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import prepare_candidate_manifest,load_candidate_manifest,normalize_prompt


class FixtureTokenizer:
    def encode(self,text,**kwargs):return list(range(len(text.split())))
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kwargs):
        pieces=[]
        for message in messages:
            content=message['content']
            if message['role']=='assistant' and '</think>' in content:content=content.split('</think>')[-1]
            pieces.append('<'+message['role']+'>'+content+'<end>')
        text=''.join(pieces)
        if add_generation_prompt:text+='<assistant><think>\n'
        return self.encode(text) if tokenize else text


def test_authorized_first_turn_exact_dedupe_overlap_and_both_caps(tmp_path):
    rows=[{'id':i,'language':lang,'conversation':[{'role':'user','content':text}]} for i,lang,text in [
        (1,'English',' A  question '),(2,'English','a question'),(3,'English','held OUT'),(4,'French','Bonjour'),(5,'English','x '*201),(6,'English','new question')]]
    tokenizers={m:FixtureTokenizer() for m in ('7b','14b')}
    result=prepare_candidate_manifest(rows,tokenizers,'a'*40,['held out'],authorized=True,output_path=tmp_path/'manifest.json')
    assert result['count']==2
    assert {r['reason'] for r in result['exclusions']}=={'normalized_duplicate','detectable_evaluation_overlap','non_english_source_label','input_exceeds_200_tokens'}
    assert load_candidate_manifest(tmp_path/'manifest.json')==result
    assert normalize_prompt('Ａ  Question')=='a question'
    assert result==prepare_candidate_manifest(rows,tokenizers,'a'*40,['held out'],authorized=True)
    with pytest.raises(PermissionError):prepare_candidate_manifest(rows,tokenizers,'a'*40,[],authorized=False)
    with pytest.raises(ValueError):prepare_candidate_manifest(rows,tokenizers,'main',[],authorized=True)


def test_official_shard_upstream_checksum_pagination_and_cache_no_hidden_retry(monkeypatch,tmp_path):
    import requests
    import huggingface_hub
    from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import acquire_pinned_parquet
    body=b'PAR1explicit checksum fixturePAR1'
    sha=content_hash(body)
    calls=[]
    class Response:
        def __init__(self,data=None,links=None):self.data,self.links=data,links or {}
        def raise_for_status(self):return None
        def json(self):return self.data
        def iter_content(self,n):yield body
        def __enter__(self):return self
        def __exit__(self,*args):return None
    class Session:
        def get(self,url,**kwargs):
            calls.append(url)
            assert kwargs.get('verify',True) is not False
            if '/resolve/' in url:return Response()
            if 'page=2' in url:return Response([{'type':'file','path':'main/test-00000-of-00001.parquet','oid':'b'*40,'lfs':{'oid':sha,'size':len(body)}}])
            return Response([{'type':'directory','path':'main'}],{'next':{'url':'https://huggingface.co/api/datasets/fixture/tree/'+('a'*40)+'?page=2'}})
    monkeypatch.setattr(requests,'Session',Session)
    monkeypatch.setattr(huggingface_hub,'get_token',lambda:None)
    artifacts=acquire_pinned_parquet('fixture','a'*40,configuration='main',split='test',cache_root=tmp_path)
    assert artifacts[0]['sha256']==sha and len(calls)==3
    assert acquire_pinned_parquet('fixture','a'*40,configuration='main',split='test',cache_root=tmp_path)==artifacts
    assert len([u for u in calls if '/resolve/' in u])==1
    __import__('pathlib').Path(artifacts[0]['path']).write_bytes(b'PAR1corruptedPAR1')
    with pytest.raises(ValueError):acquire_pinned_parquet('fixture','a'*40,configuration='main',split='test',cache_root=tmp_path)
