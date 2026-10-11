"""Explicit immutable producer-to-consumer SFT compatibility, without relabelling data."""
from pathlib import Path
from shared_procedures.configuration import configuration_hash, scientific_configuration, validate_configuration
from shared_procedures.identity import content_hash, file_hash

BINDINGS = ('src/training/sft/data_handoff.py', 'src/training/sft/sft_training.py',
            'src/training/sft/individual_fit/sft_fit.py', 'scripts/certify_sft_handoff.py', 'requirements.lock')
ROOT = Path(__file__).resolve().parents[3]


def compatibility_projection(config):
    result = scientific_configuration(dict(config))
    for key in ('implementation_revision', 'dependency_lock_hash'):
        result.get('runtime', {}).pop(key, None)
    return result


def implementation_bindings():
    paths = set(BINDINGS)
    for directory in ('src/training/sft', 'src/shared_procedures', 'src/data_generation_collection/sft_data'):
        paths.update(str(path.relative_to(ROOT)) for path in (ROOT/directory).rglob('*.py'))
    return {name: file_hash(ROOT / name) for name in sorted(paths)}


def validate_handoff(certificate, config, spec=None):
    """Recheck configuration, consumer source and exact condition identity at use."""
    if not isinstance(certificate, dict) or (certificate.get('status') != 'PASSED' or certificate.get('schema_version') != 'sft_producer_consumer_handoff_v1'):
        raise ValueError('Verified SFT data handoff required')
    if certificate.get('certificate_hash') != content_hash({k:v for k,v in certificate.items() if k != 'certificate_hash'}):
        raise ValueError('SFT handoff integrity differs')
    producer = certificate['producer_configuration']
    consumer = certificate['consumer_configuration']
    validate_configuration(consumer, runtime=True)
    if (configuration_hash(producer) != certificate['producer_configuration_hash']
            or configuration_hash(consumer) != certificate['consumer_configuration_hash']
            or configuration_hash(dict(config)) != certificate['consumer_configuration_hash']
            or compatibility_projection(producer) != compatibility_projection(consumer)
            or certificate['consumer_commit'] != consumer['runtime']['implementation_revision']
            or certificate['producer_commit'] != producer['runtime']['implementation_revision']
            or certificate['implementation_bindings'] != implementation_bindings()
            or consumer['runtime']['dependency_lock_hash'] != file_hash(ROOT/'requirements.lock')):
        raise ValueError('SFT handoff configuration/source compatibility differs')
    rows = certificate['conditions']
    expected = {(f,m,l) for f in ('rot13','kazitext') for m in ('7b','14b') for l in range(1,11)}
    if len(rows) != 40 or {(r['family'],r['model_id'],r['level']) for r in rows} != expected:
        raise ValueError('Incomplete SFT handoff matrix')
    if spec is not None:
        row = next(r for r in rows if (r['family'],r['model_id'],r['level']) == (spec.family,spec.model_id,spec.level))
        if (spec.dataset_hash != row['dataset_hash']
                or spec.producer_configuration_hash != certificate['producer_configuration_hash']
                or spec.handoff_certificate_hash != certificate['certificate_hash']
                or spec.configuration_hash != certificate['consumer_configuration_hash']):
            raise ValueError('Fit differs from certified SFT handoff')
    return certificate
