#!/usr/bin/env python3
"""Prepare or train an explicit zero-ROT13 control using the unchanged SFT engine."""
from pathlib import Path
import argparse, copy, json, os, signal, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shared_procedures.configuration import configuration_hash, validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, read_json, artifact_hook
from training.sft.individual_fit.sft_fit import SFTFitSpec, dataset_hash, validate_datasets, run_sft_fit
from data_generation_collection.sft_data.sft_data_preparation import validate_accepted_pool, load_sft_records
from experiment_overview.experiment_overview import code_revision


def prepare(pool, templates, config):
    validate_accepted_pool(pool)
    pairs={p['example_id']:p for p in pool['pairs']}
    cfg_hash=configuration_hash(config)
    result={}
    for split in ('train','validation'):
        originals=templates[split]
        if [r['example_id'] for r in originals]!=pool[split+'_ids']:
            raise ValueError('Control must retain the original ordered split')
        rows=[]
        for old in originals:
            p=pairs[old['example_id']];t=p['traces']['14b']
            if (old['clean_trace_hash']!=t['clean_trace_hash'] or old['prompt']!=p['prompt']
                    or old['final_answer']!=t['final_answer'] or old['model_revision']!=t['model_revision']):
                raise ValueError('Control source differs from certified encoded source')
            r=copy.deepcopy(old)
            r.update(level=0, method='rot13', family='rot13', config_hash=cfg_hash,
                     reasoning=t['reasoning'], completion='<think>'+t['reasoning']+'</think>'+t['final_answer'],
                     transformation={'version':'unencoded_control_v1','encoding_fraction':0.0},
                     transformed_token_count=t['original_token_count'])
            r['identity']={'version':'unencoded_control_v1','source_record_hash':content_hash(old),
                           'source_pool_hash':pool['content_hash'],'configuration_hash':cfg_hash}
            rows.append(r)
        result[split]=rows
    model=config['models']['14b']
    spec=SFTFitSpec('rot13','14b',0,model['revision'],model['tokenizer_revision'],cfg_hash,
                    dataset_hash(result['train'],result['validation']),training_budget='efficient_one_epoch_v1')
    validate_datasets(spec,result['train'],result['validation'])
    return spec,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--pool');parser.add_argument('--template-root')
    parser.add_argument('--config',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--output');parser.add_argument('--previous-preflight')
    args=parser.parse_args()
    config=validate_configuration(read_json(args.config),runtime=True)
    if config['sft']['selection']!={'models':['14b'],'families':['rot13'],'levels':[0]}:
        raise ValueError('Explicit 14B level-zero selection required')
    if config['runtime']['implementation_revision']!=code_revision():
        raise ValueError('Executed source differs from declared control source')
    data=Path(args.data)
    if args.prepare:
        pool=read_json(args.pool)
        original_manifest=read_json(Path(args.template_root)/'manifest.json')
        if original_manifest.get('status')!='verified' or original_manifest.get('simulation') is not False:
            raise ValueError('Verified production source manifest required')
        for split in ('train','validation'):
            if file_hash(Path(args.template_root)/(split+'.jsonl'))!=original_manifest['content_hashes'][split]:
                raise ValueError('Certified template bytes changed')
        templates={s:load_sft_records(Path(args.template_root)/(s+'.jsonl')) for s in ('train','validation')}
        spec,rows=prepare(pool,templates,config);data.mkdir(parents=True,exist_ok=True)
        manifest={'version':'unencoded_control_v1','spec':spec.identity(),'pool_hash':pool['content_hash'],
                  'source_pool_file_sha256':file_hash(args.pool),'files':{},'source_commit':code_revision()}
        for split in rows:
            p=data/(split+'.jsonl')
            if p.exists():raise ValueError('Preserving existing prepared control')
            with p.open('w') as f:
                for r in rows[split]:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            manifest['files'][split]={'sha256':file_hash(p),'count':len(rows[split])}
        manifest['manifest_hash']=content_hash(manifest);atomic_write_json(data/'manifest.json',manifest)
        print(json.dumps({'status':'PREPARED','counts':manifest['files'],'dataset_hash':spec.dataset_hash}),flush=True)
        return
    from operations.artifacts import ArtifactService
    from operations.drive import DriveStore,drive_service
    from shared_procedures.preflight import CheckResult,run_preflights,require_preflights
    from shared_procedures.recovery_procedure import configure_worker_recovery
    from data_generation_collection.evaluation_data.protocol.analysis_plan.statistical_validation import implementation_hash
    manifest=read_json(data/'manifest.json')
    if manifest['manifest_hash']!=content_hash({k:v for k,v in manifest.items() if k!='manifest_hash'}):raise ValueError('Control manifest changed')
    spec=SFTFitSpec.from_dict(manifest['spec'])
    if spec.configuration_hash!=configuration_hash(config):raise ValueError('Control configuration changed')
    for split in ('train','validation'):
        if file_hash(data/(split+'.jsonl'))!=manifest['files'][split]['sha256']:raise ValueError('Control bytes changed')
    rows={s:load_sft_records(data/(s+'.jsonl')) for s in ('train','validation')}
    validate_datasets(spec,rows['train'],rows['validation'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    configure_worker_recovery(output,worker_id='rot13_14b_unencoded_control')
    service=ArtifactService(output,output/'drive_registry.json',DriveStore(drive_service(os.environ['GOOGLE_OAUTH_TOKEN_FILE']),os.environ['GOOGLE_DRIVE_FOLDER_ID']),drive_recovery=True)
    provenance={'experiment_id':spec.configuration_hash,'branch':'rot13_control','model':'14b','run_id':spec.run_id,
                'stage':'sft','producer':'run_unencoded_sft_control','source_code_commit':code_revision(),
                'configuration_hash':spec.configuration_hash,'dataset_revision':spec.dataset_hash,
                'model_revision':spec.model_revision,'seeds':[42]}
    preserve=service.completion_hook(provenance)
    probe=output/'launch_provenance.json';atomic_write_json(probe,{'source_commit':code_revision(),'data_manifest':manifest,'configuration':config})
    preserve(probe,{'kind':'control_launch'})
    if service.sync_artifacts()['status']!='VERIFIED':raise RuntimeError('Drive publication unavailable')
    previous=read_json(args.previous_preflight)
    if previous['report_hash']!=content_hash({k:v for k,v in previous.items() if k!='report_hash'}):raise ValueError('Prior report hash mismatch')
    historical=next(c for c in previous['checks'] if c['name']=='durable_storage')
    if historical['status']!='passed':raise ValueError('No verified historical durability evidence')
    statistical=read_json(ROOT/'docs/evidence/statistical_guard_v2_report.json')
    if statistical['report_hash']!=content_hash({k:v for k,v in statistical.items() if k!='report_hash'}) or statistical['implementation_hash']!=implementation_hash():raise ValueError('Statistical evidence no longer compatible')
    stat=copy.deepcopy(statistical['checks'][0]);stat['evidence']['compatible_reuse']={'source_report_hash':statistical['report_hash'],'consumer_commit':code_revision()}
    checks={'clean_data':CheckResult('clean_data','passed','Original certified sources restored without encoding; ordered splits, clean trace hashes and exact payload verified.',{'live':True,'simulation':False,'manifest_hash':manifest['manifest_hash']}),
            'durable_storage':CheckResult('durable_storage','passed','Live Drive publication verified; prior independent transport readback retained.',{'live':True,'previous_report_hash':previous['report_hash'],'historical_readback':historical['evidence']}),
            'statistical_validation':CheckResult(**stat)}
    report=run_preflights(config,checks,output_dir=output/'preflight',scope='published_sft')
    require_preflights(report,['clean_data','durable_storage','spending_authorization','statistical_validation'])
    stop={'requested':False}
    def pause(*_):stop['requested']=True
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    with artifact_hook(preserve):
        result=run_sft_fit(spec,rows['train'],rows['validation'],output,config,preflight_report=report,should_pause=lambda:stop['requested'])
    if service.sync_artifacts()['status']!='VERIFIED':raise RuntimeError('Checkpoint publication incomplete')
    service.export_manifest()
    print(json.dumps({'status':result['status'],'progress':result['progress'],'drive_status':'VERIFIED'}),flush=True)

if __name__=='__main__':main()
