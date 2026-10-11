"""Tampering and scientific changes must not cross the producer/consumer boundary."""
from copy import deepcopy
import pytest
from shared_procedures.configuration import FIXED, configuration_hash
from shared_procedures.identity import content_hash, file_hash
from training.sft.data_handoff import ROOT, compatibility_projection, implementation_bindings, validate_handoff
from training.sft.individual_fit.sft_fit import MODELS, SFTFitSpec


def certificate_fixture():
    producer = {}
    for key, value in FIXED.items():
        target = producer
        parts = key.split('.')
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = deepcopy(value)
    producer['models'] = {size:{'id':name,'revision':'a'*40,'tokenizer_revision':'b'*40} for size,name in MODELS.items()}
    for source in producer['datasets'].values():
        source['revision'] = 'd'*40
    producer['interventions']['kazitext'].update(udpipe_sha256='e'*64,
        aspell_dictionary_provenance={'settings':{'language':'en'},'files':[{'name':'dictionary','sha256':'f'*64}]})
    producer['runtime'] = {'implementation_revision':'1'*40}
    consumer = deepcopy(producer)
    consumer['runtime'].update(implementation_revision='2'*40,dependency_lock_hash=file_hash(ROOT/'requirements.lock'))
    certificate = {'schema_version':'sft_producer_consumer_handoff_v1','status':'PASSED',
        'producer_configuration':producer,'consumer_configuration':consumer,
        'producer_configuration_hash':configuration_hash(producer),'consumer_configuration_hash':configuration_hash(consumer),
        'producer_commit':'1'*40,'consumer_commit':'2'*40,'implementation_bindings':implementation_bindings(),
        'conditions':[{'family':f,'model_id':m,'level':l,'dataset_hash':'9'*64}
                      for f in ('rot13','kazitext') for m in MODELS for l in range(1,11)]}
    certificate['certificate_hash'] = content_hash(certificate)
    return certificate, consumer


def resign(certificate):
    certificate['certificate_hash'] = content_hash({k:v for k,v in certificate.items() if k != 'certificate_hash'})


def test_consumer_is_explicit_and_original_producer_identity_is_preserved():
    certificate, config = certificate_fixture()
    spec = SFTFitSpec('rot13','7b',1,'a'*40,'b'*40,configuration_hash(config),'9'*64,
                      certificate['producer_configuration_hash'],certificate['certificate_hash'])
    assert validate_handoff(certificate,config,spec) is certificate
    assert compatibility_projection(config) == compatibility_projection(certificate['producer_configuration'])
    assert spec.producer_configuration_hash != spec.configuration_hash
    with pytest.raises(ValueError,match='Fit differs'):
        validate_handoff(certificate,config,SFTFitSpec('rot13','7b',1,'a'*40,'b'*40,configuration_hash(config),'8'*64,
                         certificate['producer_configuration_hash'],certificate['certificate_hash']))


@pytest.mark.parametrize('mutation',['scientific','source','matrix','integrity','configuration'])
def test_handoff_rejects_changes_even_with_rehashed_certificate(mutation):
    certificate, config = certificate_fixture()
    if mutation == 'scientific':
        certificate['producer_configuration']['generation']['max_new_tokens'] = 100
    elif mutation == 'source':
        certificate['implementation_bindings']['requirements.lock'] = '0'*64
    elif mutation == 'matrix':
        certificate['conditions'].pop()
    elif mutation == 'integrity':
        certificate['status'] = 'FAKE'
    else:
        config['runtime']['implementation_revision'] = '3'*40
    resign(certificate)
    with pytest.raises(ValueError):
        validate_handoff(certificate,config)
