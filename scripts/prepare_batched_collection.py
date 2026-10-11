#!/usr/bin/env python3
"""Rebind unchanged audited inputs and adopt verified whole raw responses.

Original files and producer records remain immutable. New raw-cache copies
explicitly identify their original producer and Drive artifact; this does not
claim that existing responses were generated with the new batching backend.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from operations.artifacts import ArtifactService, validate_registered_record
from operations.collection_inputs import validate_consumer_certificate
from operations.drive import file_hash
from shared_procedures.configuration import load_configuration, configuration_hash, scientific_configuration
from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.storage import atomic_write_json, read_json
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import GENERATION_SETTINGS


def unchanged_science(old, new):
    left, right = scientific_configuration(old), scientific_configuration(new)
    for value in (left, right):
        value['runtime'].pop('implementation_revision', None)
    if left != right:
        raise RuntimeError('Input reuse requires identical scientific settings except the implementation revision')


def prepare(old, new, *, old_certificate, input_root, certificate_output, old_registry,
            old_data_root, new_data_root, report_output):
    unchanged_science(old, new)
    old_commit, new_commit = (config['runtime']['implementation_revision'] for config in (old, new))
    actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    if actual != new_commit:
        raise RuntimeError('New consumer configuration must match the clean executed checkout')
    old_hash, new_hash = configuration_hash(old), configuration_hash(new)
    certificate = validate_consumer_certificate(old_certificate,old,consumer_commit=old_commit,
        input_root=input_root,asset_cache=new['runtime']['cache_root'])
    rebound = deepcopy(certificate)
    rebound.update(consumer_commit=new_commit,consumer_configuration_hash=new_hash,
        consumer_validation_mode='verified_unchanged_input_audit_rebinding',
        audit_parent_consumer_commit=old_commit,audit_parent_configuration_hash=old_hash,
        audit_parent_certificate_sha256=file_hash(Path(old_certificate)))
    rebound['certificate_hash']=content_hash({k:v for k,v in rebound.items() if k!='certificate_hash'})
    target=Path(certificate_output)
    if target.exists():
        if read_json(target)!=rebound:raise RuntimeError('Existing rebound certificate differs')
    else:atomic_write_json(target,rebound)
    validate_consumer_certificate(target,new,consumer_commit=new_commit,input_root=input_root,
                                  asset_cache=new['runtime']['cache_root'])
    registry=ArtifactService(Path(old_data_root).parent,old_registry)
    originals=registry.list_artifacts()
    adopted=[]
    for record in originals:
        if '/clean_traces/raw/production/' not in record['relative_path']:
            continue
        validate_registered_record(record)
        if (record['sync_status']!='VERIFIED' or not record.get('drive_file_id')
            or record['source_code_commit']!=old_commit or record['configuration_hash']!=old_hash):
            raise RuntimeError('Original whole responses must be verified in Drive under the original producer')
        source=Path(record['local_path'])
        if source.is_symlink() or not source.resolve().is_relative_to(Path(old_data_root).resolve()):
            raise RuntimeError('Original cache is outside its declared data root')
        if source.stat().st_size!=record['size'] or file_hash(source)!=record['sha256']:
            raise RuntimeError('Original cache bytes differ from the verified producer record')
        raw=read_json(source)
        model=raw['model_id'];spec=new['models'][model]
        if (raw.get('config_hash')!=old_hash or raw.get('raw_hash')!=content_hash(raw.get('raw_generation'))
            or raw.get('generation_settings')!=GENERATION_SETTINGS or raw.get('simulation') is not False
            or raw.get('execution_mode')!='production' or raw.get('model_name')!=spec['id']
            or raw.get('model_revision')!=spec['revision'] or raw.get('tokenizer_revision')!=spec['tokenizer_revision']
            or raw.get('seed')!=derive_seed(42,'clean_generation',raw['example_id'],model)
            or type(raw.get('generated_token_count')) is not int
            or raw['generated_token_count']!=len(raw.get('generation_token_ids',[]))
            or not 0<raw['generated_token_count']<=3072 or raw.get('finish_reason') not in ('stop','length')):
            raise RuntimeError('Original response generation identity or frozen decoding policy differs')
        adopted_raw={**raw,'config_hash':new_hash,'generation_original_producer_commit':old_commit,
            'generation_original_configuration_hash':old_hash,'generation_original_file_sha256':record['sha256'],
            'generation_original_artifact_id':record['artifact_id'],'generation_original_drive_file_id':record['drive_file_id'],
            'cache_adoption_mode':'original_whole_response_with_explicit_consumer_binding'}
        if raw.get('generation_original_producer_commit'):
            adopted_raw['cache_adoption_history']=[*raw.get('cache_adoption_history',[]),
                {key:raw[key] for key in raw if key.startswith('generation_original_')}]
        destination=Path(new_data_root)/source.relative_to(Path(old_data_root))
        if destination.exists():
            if read_json(destination)!=adopted_raw:raise RuntimeError('Existing adopted response differs')
        else:atomic_write_json(destination,adopted_raw)
        adopted.append({'model':model,'example_id':raw['example_id'],'destination':str(destination),
                        'original_drive_file_id':record['drive_file_id'],'original_sha256':record['sha256']})
    report={'status':'VERIFIED_UNCHANGED_INPUTS_AND_ORIGINAL_WHOLE_RESPONSE_ADOPTION',
        'old_commit':old_commit,'new_commit':new_commit,'old_configuration_hash':old_hash,
        'new_configuration_hash':new_hash,'original_files_unchanged':True,
        'new_model_outputs_generated':False,'audit_parent_certificate_sha256':file_hash(Path(old_certificate)),
        'rebound_certificate':str(target),'adopted_response_count':len(adopted),'adopted':adopted}
    atomic_write_json(report_output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old-config','new-config','old-certificate','input-root','certificate-output',
                 'old-registry','old-data-root','new-data-root','report-output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    # Private root variables must not rewrite the original producer configuration.
    import os
    saved={key:os.environ.pop(key,None) for key in ('DATA_ROOT','ARTIFACT_ROOT','MODEL_CACHE_ROOT')}
    try:old,new=load_configuration(args.old_config),load_configuration(args.new_config)
    finally:
        for key,value in saved.items():
            if value is not None:os.environ[key]=value
    report=prepare(old,new,old_certificate=args.old_certificate,input_root=args.input_root,
        certificate_output=args.certificate_output,old_registry=args.old_registry,
        old_data_root=args.old_data_root,new_data_root=args.new_data_root,report_output=args.report_output)
    print(json.dumps({key:report[key] for key in ('status','adopted_response_count','new_commit','new_configuration_hash')}))


if __name__=='__main__':main()
