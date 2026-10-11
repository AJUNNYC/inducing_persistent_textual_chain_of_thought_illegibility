#!/usr/bin/env python3
"""Certify the existing Drive-audited corpus for an explicitly frozen consumer.

Reuses completed Drive readback; verifies current local bytes and production
worklist invariants. No generation, transformation, GPU or remote Pod required.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shared_procedures.configuration import configuration_hash, validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json
from training.sft.data_handoff import compatibility_projection, implementation_bindings, validate_handoff
from training.sft.sft_training import build_worklist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('producer-config', 'consumer-config', 'audit', 'bundle', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    producer = json.loads(Path(args.producer_config).read_text())
    consumer = json.loads(Path(args.consumer_config).read_text())
    validate_configuration(consumer, runtime=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if consumer['runtime']['implementation_revision'] != commit:
        raise ValueError('Consumer must name the actual frozen source commit')
    if compatibility_projection(producer) != compatibility_projection(consumer):
        raise ValueError('Scientific settings changed between producer and consumer')
    audit = json.loads(Path(args.audit).read_text())
    bundle = json.loads(Path(args.bundle).read_text())
    if (audit['status'] != 'PASSED' or audit['verified_files'] != 122
            or audit['validation']['record_count'] != 404000
            or audit['generation_commit'] != producer['runtime']['implementation_revision']
            or bundle['config_hash'] != configuration_hash(producer)):
        raise ValueError('Original Drive readback/source binding differs')
    for entry in audit['artifacts']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or not str(relative).startswith('data/vllm_sft_preparation/'):
            raise ValueError('Unsafe audit path')
        path = Path('/workspace')/relative
        if path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
            raise ValueError('Current corpus bytes differ from Drive readback')
    if Path(args.bundle).resolve() != Path('/workspace/data/vllm_sft_preparation/transformed/manifest.json'):
        raise ValueError('Bundle must be the independently audited corpus manifest')
    print('122 corpus artifacts match completed Drive readback', flush=True)
    tasks = build_worklist(producer, bundle['datasets'])
    certificate = {
        'schema_version':'sft_producer_consumer_handoff_v1', 'status':'PASSED',
        'checked_at':datetime.now(timezone.utc).isoformat(),
        'producer_configuration':producer, 'consumer_configuration':consumer,
        'producer_configuration_hash':configuration_hash(producer),
        'consumer_configuration_hash':configuration_hash(consumer),
        'producer_commit':producer['runtime']['implementation_revision'], 'consumer_commit':commit,
        'implementation_bindings':implementation_bindings(),
        'drive_readback_report_sha256':file_hash(args.audit),
        'drive_index_id':audit['index_file_id'], 'bundle_hash':bundle['content_hash'],
        'verified_artifacts':audit['artifacts'], 'record_count':404000,
        'conditions':[{k:t['spec'][k] for k in ('family','model_id','level','dataset_hash')} for t in tasks],
        'scientific_settings_unchanged':True, 'producer_records_modified':False,
        'scope':'Data compatibility only; does not replace baseline repeatability or launch preflights.'}
    certificate['certificate_hash'] = content_hash(certificate)
    validate_handoff(certificate, consumer)
    atomic_write_json(Path(args.output)/'certificate.json', certificate)
    # Specs preserve original producer hashes while naming the actual consumer.
    for task in tasks:
        task['spec'].update(configuration_hash=certificate['consumer_configuration_hash'],
                            producer_configuration_hash=certificate['producer_configuration_hash'],
                            handoff_certificate_hash=certificate['certificate_hash'])
        from training.sft.individual_fit.sft_fit import SFTFitSpec
        validate_handoff(certificate, consumer, SFTFitSpec.from_dict(task['spec']))
    atomic_write_json(Path(args.output)/'worklist.json', tasks)
    print(json.dumps({'status':'PASSED','conditions':len(tasks),'records':404000,
                      'certificate_hash':certificate['certificate_hash']}),flush=True)

if __name__ == '__main__':
    main()
