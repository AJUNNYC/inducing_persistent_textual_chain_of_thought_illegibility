from copy import deepcopy
from pathlib import Path
import pytest
from shared_procedures.configuration import *
CONFIG=Path(__file__).resolve().parents[3]/'config.yaml'

def test_config_fixed_science_and_matrix():
    c=load_configuration(CONFIG)
    assert len(sft_matrix(c))==40
    assert sum(c['evaluation']['capability_counts'].values())==1233
    assert len(c['interventions']['kazitext']['alphas'])==10

@pytest.mark.parametrize('field,value',[('seed',43),('epochs',1),('panel',1240),('model','Qwen3')])
def test_scientific_mutation_rejected(field,value):
    c=load_configuration(CONFIG)
    if field=='seed':c['seed']=value
    elif field=='epochs':c['sft']['epochs']=value
    elif field=='panel':c['evaluation']['capability_counts']['arc']=value
    else:c['models']['7b']['id']=value
    with pytest.raises(ConfigurationError):validate_configuration(c)

def test_unresolved_runtime_not_frozen(tmp_path):
    c=load_configuration(CONFIG);c['models']['7b']['revision']=None
    with pytest.raises(ConfigurationError):freeze_configuration(c,tmp_path/'freeze.json')
    assert not (tmp_path/'freeze.json').exists()

def test_duplicate_yaml_and_credentials_rejected(tmp_path):
    path=tmp_path/'bad.yaml';path.write_text('seed: 42\nseed: 43\n')
    with pytest.raises(ConfigurationError):load_configuration(path)
    c=load_configuration(CONFIG);c['runtime']['api_key']='forbidden'
    with pytest.raises(ConfigurationError):validate_configuration(c)

def test_runtime_env_overrides_and_private_values(monkeypatch,tmp_path):
    monkeypatch.setenv('DATA_ROOT',str(tmp_path/'data'))
    config=load_configuration(CONFIG)
    assert config['runtime']['data_root']==str(tmp_path/'data')
    env=tmp_path/'.env';env.write_text('MODEL_OPENAI_API_KEY="private test value"\n# never expand shell\nSHELL_TEXT=\'$(echo unsafe)\'\n')
    monkeypatch.delenv('MODEL_OPENAI_API_KEY',raising=False);monkeypatch.delenv('SHELL_TEXT',raising=False)
    load_private_environment(env)
    import os
    assert os.environ['MODEL_OPENAI_API_KEY']=='private test value' and os.environ['SHELL_TEXT']=='$(echo unsafe)'
    assert 'private test value' not in str(config)

def test_scientific_hash_excludes_only_operational_changes():
    c=load_configuration(CONFIG);changed=deepcopy(c)
    changed['runtime']['artifact_root']='/another/path';changed['runtime']['credential_env']={'openai':['PRIVATE_BINDING_NAME']}
    changed['spending'].update(authorized=True,authorized_usd=500,authorization_evidence='test',provider_prices={'new':'rates'})
    changed['judges']['quota_limits']={'openai':{'requests_per_minute':3,'tokens_per_minute':500}}
    assert configuration_hash(c)==configuration_hash(changed)
    assert snapshot_hash(c)!=snapshot_hash(changed)
    changed['runtime']['implementation_revision']='a'*40
    assert configuration_hash(c)!=configuration_hash(changed)
    changed=deepcopy(c);changed['operations']['mbpp_image']='python@sha256:'+'b'*64
    assert configuration_hash(c)!=configuration_hash(changed)

def test_operational_snapshots_append_without_rewriting_science_freeze(tmp_path):
    c=load_configuration(CONFIG);c['interventions']['kazitext']['udpipe_sha256']='d'*64
    c['runtime'].update(implementation_revision='e'*40,dependency_lock_hash='f'*64)
    first=freeze_configuration(c,tmp_path/'frozen.json')
    from shared_procedures.storage import read_json
    canonical=read_json(tmp_path/'frozen.json')
    c['spending'].update(authorized=True,authorized_usd=100,authorization_evidence='fixture authorization')
    second=freeze_configuration(c,tmp_path/'frozen.json')
    assert first['configuration_hash']==second['configuration_hash'] and first['snapshot_hash']!=second['snapshot_hash']
    assert read_json(tmp_path/'frozen.json')==canonical
    assert len(list(tmp_path.glob('frozen_snapshot_*.json')))==2

def test_local_udpipe_dictionary_paths_excluded_but_content_provenance_retained():
    c=load_configuration(CONFIG);changed=deepcopy(c)
    changed['operations'].update(udpipe_path='/new/model.udpipe',aspell_dictionary_files=['/new/en_dictionary'])
    assert configuration_hash(c)==configuration_hash(changed)
    changed['interventions']['kazitext']['aspell_dictionary_provenance']=[{'name':'en_dictionary','sha256':'a'*64}]
    assert configuration_hash(c)!=configuration_hash(changed)

def test_drive_primary_recovery_evidence_is_operational_and_snapshot_audited():
    c=load_configuration(CONFIG);changed=deepcopy(c)
    changed['runtime']['durability_evidence']={'mode':'google_drive_primary','live':True,
        'registered_probe':{'sha256':'a'*64},'independent_restore':{'observed_at':'explicit_fixture'}}
    assert configuration_hash(c)==configuration_hash(changed)
    assert snapshot_hash(c)!=snapshot_hash(changed)

@pytest.mark.parametrize('mutation',['missing','duplicate','path','bad_hash'])
def test_incomplete_aspell_provenance_prevents_runtime_freeze(tmp_path,mutation):
    c=load_configuration(CONFIG)
    c['runtime'].update(implementation_revision='e'*40,dependency_lock_hash='f'*64)
    entries=c['interventions']['kazitext']['aspell_dictionary_provenance']['files']
    if mutation=='missing':c['interventions']['kazitext'].pop('aspell_dictionary_provenance')
    elif mutation=='duplicate':entries.append(deepcopy(entries[0]))
    elif mutation=='path':entries[0]['name']='/machine/local/path'
    else:entries[0]['sha256']='unverified'
    with pytest.raises(ConfigurationError,match='aspell_dictionary_provenance'):
        freeze_configuration(c,tmp_path/'freeze.json')
    assert not (tmp_path/'freeze.json').exists()
