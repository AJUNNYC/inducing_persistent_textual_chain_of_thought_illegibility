"""Explicit immutable producer-to-consumer SFT compatibility, without relabelling data."""
from pathlib import Path
from shared_procedures.configuration import (configuration_hash, scientific_configuration, validate_configuration,
    sft_training_budget, sft_budget_epochs, sft_execution_selection, SFT_BUDGET_OVERRIDES)
from shared_procedures.identity import content_hash, file_hash

BINDINGS = ('src/training/sft/data_handoff.py', 'src/training/sft/sft_training.py',
            'src/training/sft/individual_fit/sft_fit.py', 'scripts/certify_sft_handoff.py', 'requirements.lock')
ROOT = Path(__file__).resolve().parents[3]


def compatibility_projection(config):
    """Corpus compatibility; the explicit training amendment is checked separately."""
    result = scientific_configuration(dict(config))
    for key in ('implementation_revision', 'dependency_lock_hash'):
        result.get('runtime', {}).pop(key, None)
    for key in ('training_budget', 'epochs', 'selection'):
        result['sft'].pop(key, None)
    # Configuration validation above proves the exact named recipe. Normalize
    # its declared training-only changes back to the original producer settings.
    from shared_procedures.configuration import FIXED
    for key in {key for values in SFT_BUDGET_OVERRIDES.values() for key in values}:
        result['sft'][key.split('.', 1)[1]] = FIXED[key]
    return result


def training_amendment(producer, consumer):
    """Record the approved budget/scope change without relabelling producer data."""
    # Preserve the exact amendment representation in historical certificates.
    # Only the revised published recipe declares the new physical batching.
    batching_revision = (sft_training_budget(consumer) == 'efficient_one_epoch_v1'
                         and consumer['sft'].get('microbatch_size') == 2
                         and consumer['sft'].get('gradient_accumulation_steps') == 8)
    def settings(config):
        names = ('quantization', 'load_in_4bit', 'double_quantization', 'optimizer',
                 'gradient_checkpointing')
        if batching_revision:
            names += ('microbatch_size', 'gradient_accumulation_steps')
        return {'training_budget': sft_training_budget(config), 'epochs': sft_budget_epochs(config),
                'recipe_changes': {name: config['sft'][name] for name in names},
                'selected_conditions': sft_execution_selection(config)}
    before, after = settings(producer), settings(consumer)
    if before == after:
        return None
    return {'schema_version': 'sft_training_budget_scope_amendment_v1',
            'before': before, 'after': after,
            'unchanged': ['producer_record_bytes', 'model_and_tokenizer_revisions',
                          'completion_only_loss', 'full_response_lengths', 'train_validation_partitions',
                          'lora_settings_and_learning_rate'],
            'interpretation': 'Declared shorter published training recipe; not the original two-epoch experiment.'}


def implementation_bindings():
    paths = set(BINDINGS)
    if (ROOT/'requirements_sft_published.lock').is_file():
        paths.add('requirements_sft_published.lock')
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
    amendment = training_amendment(producer, consumer)
    if amendment is not None:
        if (certificate.get('consumer_training_amendment') != amendment
                or certificate.get('scientific_settings_unchanged') is not False
                or certificate.get('corpus_scientific_settings_unchanged') is not True
                or certificate.get('producer_records_modified') is not False
                or certificate.get('selected_conditions') != sft_execution_selection(consumer)):
            raise ValueError('Explicit SFT budget/scope amendment differs or is missing')
    elif certificate.get('consumer_training_amendment') is not None:
        raise ValueError('Unexpected SFT budget/scope amendment')
    if ('selected_conditions' in certificate
            and certificate['selected_conditions'] != sft_execution_selection(consumer)):
        raise ValueError('Certified SFT execution selection differs')
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
                or spec.configuration_hash != certificate['consumer_configuration_hash']
                or spec.training_budget != sft_training_budget(consumer)):
            raise ValueError('Fit differs from certified SFT handoff')
    return certificate
