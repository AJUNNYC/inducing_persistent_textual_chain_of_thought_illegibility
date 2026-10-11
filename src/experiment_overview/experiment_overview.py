"""Agent-accessible execution of frozen first-stage scientific and operational tasks."""
from __future__ import annotations
import argparse
from copy import deepcopy
import re
import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from shared_procedures.configuration import load_configuration, configuration_hash, validate_configuration, freeze_configuration, scientific_configuration
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, artifact_hook, notify_artifact, file_lock, fsync_directory
from operations.workflow import StageRunner, experiment_graph, topological_order

ROOT = Path(__file__).resolve().parents[2]


def code_revision() -> str:
    supplied = os.environ.get('SOURCE_CODE_COMMIT')
    if supplied and not re.fullmatch(r'[0-9a-f]{40}', supplied):
        raise ValueError('SOURCE_CODE_COMMIT must identify an immutable Git commit')
    value = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=False)
    if value.returncode == 0:
        actual=value.stdout.strip()
        if supplied and actual!=supplied:
            raise RuntimeError('Declared image/source commit differs from the actual checkout')
        return actual
    if supplied:
        return supplied
    # Discovery remains usable in an image without .git. Paid scientific
    # execution requires an explicitly pinned build commit, checked below.
    return 'unversioned_' + content_hash({str(p.relative_to(ROOT)):content_hash(p.read_bytes())
                                        for p in sorted((ROOT/'src').rglob('*.py'))})


def drive_store(config):
    from operations.drive import DriveStore, drive_service
    token = os.environ.get('GOOGLE_OAUTH_TOKEN_FILE')
    if not token:
        raise RuntimeError('GOOGLE_OAUTH_TOKEN_FILE is required; authorize the existing desktop OAuth client')
    folder = os.environ.get('GOOGLE_DRIVE_FOLDER_ID') or config.get('operations',{}).get('drive_folder_id')
    return DriveStore(drive_service(token),folder)


class Experiment:
    def __init__(self, config, *, config_path='config.yaml'):
        self.config, self.config_path = deepcopy(config), str(Path(config_path).resolve())
        base = Path(self.config_path).parent
        for field, variable in (('data_root','DATA_ROOT'),('artifact_root','ARTIFACT_ROOT'),('cache_root','MODEL_CACHE_ROOT')):
            selected = Path(os.environ.get(variable) or self.config['runtime'][field])
            self.config['runtime'][field] = str((base/selected if not selected.is_absolute() else selected).resolve())
            os.environ[variable]=self.config['runtime'][field]
        self.data = Path(self.config['runtime']['data_root'])
        self.artifacts = Path(self.config['runtime']['artifact_root'])
        self.cache = Path(self.config['runtime']['cache_root'])
        # All workers use one standard Hub cache populated by the bounded,
        # checksummed provisioner. Set these before importing Hub/Transformers.
        os.environ['HF_HOME'] = str(self.cache)
        os.environ['HF_HUB_CACHE'] = str(self.cache/'hub')
        os.environ['HF_DATASETS_CACHE'] = str(self.cache/'datasets_sdk')
        self.data.mkdir(parents=True,exist_ok=True)
        self.artifacts.mkdir(parents=True,exist_ok=True)
        self.hash = configuration_hash(self.config)
        self.code = code_revision()
        self.runner = StageRunner(experiment_graph(),self.artifacts/'state'/'workflow',self.hash,self.code)
        self._judge = None
        self._artifact_service = None
        self._current_stage = None

    def write(self, path, value, status='VERIFIED'):
        atomic_write_json(path,value)
        return {'status':status,'output_files':[str(Path(path).resolve())]}

    def load(self, path):
        return read_json(path)

    def sandbox(self):
        from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.mbpp_sandbox import DockerMBPPSandbox
        image = self.config.get('operations',{}).get('mbpp_image') or 'python@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce'
        return DockerMBPPSandbox(image)

    def judge(self):
        if self._judge is None:
            from shared_procedures.judge_client import JudgeClient
            from shared_procedures.cost_accounting import CostLedger
            spend = self.config['spending']
            if not spend.get('authorized') or not spend.get('authorization_evidence'):
                raise RuntimeError('Current paid API authorization and account price/quotas evidence required')
            ledger = CostLedger(self.artifacts/'state'/'spend.json',authorization=spend)
            self._judge = JudgeClient(self.artifacts/'judge_cache',self.artifacts/'state',cost_ledger=ledger,
                         provider_prices=spend.get('provider_prices'),pricing_evidence=spend.get('pricing_evidence'),
                         quota_limits=self.config['judges'].get('quota_limits'),quota_evidence=self.config['judges'].get('quota_evidence'))
        return self._judge

    def tokenizers(self):
        from transformers import AutoTokenizer
        from operations.assets import model_load_kwargs
        for model in self.config['models'].values():
            if not isinstance(model.get('tokenizer_revision'),str) or not re.fullmatch(r'[0-9a-f]{40}',model['tokenizer_revision']):
                raise RuntimeError('Immutable construction tokenizer revisions are required')
        return {k:AutoTokenizer.from_pretrained(v['id'],revision=v['tokenizer_revision'],use_fast=True,trust_remote_code=False,
                    **model_load_kwargs(v['id'],v['tokenizer_revision'],self.cache,tokenizer_only=True)) for k,v in self.config['models'].items()}

    def kazi(self):
        from data_generation_collection.sft_data.kazitext.kazitext_data_construction import OfficialKaziText, provision_kazitext_assets
        options = self.config.get('operations',{})
        assets = provision_kazitext_assets(self.cache/'kazitext_auxiliary',source_dir=self.cache/'kazitext',udpipe_path=options.get('udpipe_path'))
        backend=OfficialKaziText(assets['source_dir'],assets['udpipe_path'],expected_udpipe_sha256=self.config['interventions']['kazitext']['udpipe_sha256'],aspell_dictionary_files=assets['aspell_dictionary_files'])
        actual={'files':backend.provenance['aspell_dictionaries'],'settings':backend.provenance['aspell_settings']}
        expected=self.config['interventions']['kazitext']['aspell_dictionary_provenance']
        if actual!={'files':expected['files'],'settings':expected['settings']}:
            backend.close()
            raise RuntimeError('Native dictionary content/settings differ from the frozen scientific configuration')
        return backend

    def source_pools(self):
        from data_generation_collection.evaluation_data.capability_panel.capability_panel_construction import load_evaluation_sources
        # This wrapper is distinct from the plain upstream source export used
        # by native sandbox certification and panel construction.
        path=self.data/'evaluation'/'official_pools_cache.json'
        if path.exists():
            record=self.load(path)
            if record['revisions'] != {k:v['revision'] for k,v in self.config['datasets'].items() if k!='lmsys'}:
                raise RuntimeError('Cached official dataset revisions changed')
            if record.get('content_hash') != content_hash(record['pools']):
                raise RuntimeError('Official source cache content changed')
            return record['pools']
        revisions={k:v['revision'] for k,v in self.config['datasets'].items() if k!='lmsys'}
        pools=load_evaluation_sources(revisions,cache_root=self.cache)
        self.write(path,{'revisions':revisions,'pools':pools,'content_hash':content_hash(pools)})
        return pools

    def protocol(self, *, trained=False):
        package=self.load(self.data/'evaluation'/'evaluation_package.json')
        if trained:
            from data_generation_collection.evaluation_data.protocol.evaluation_protocol import build_evaluation_protocol
            plan=self.load(self.data/'evaluation'/'analysis_plan.json')
            preflight=self.load(self.artifacts/'preflight'/'report.json')
            return build_evaluation_protocol(package['capability_panel'],package['legibility_subset'],analysis_plan=plan,preflight_reports=preflight,configuration_hash=self.hash,require_ready=True)
        return package['protocol']

    def state(self, stage):
        model=self.config['models'][stage.model]
        state={'model':stage.model,'model_revision':model['revision'],'tokenizer_revision':model['tokenizer_revision'],
               'runtime_hash':content_hash(scientific_configuration(self.config).get('runtime',{})),'cache_root':str(self.cache),'family':'base','state_id':f'base_{stage.model}',
               'checkpoint_hash':content_hash(model),'configuration_hash':self.hash}
        if stage.branch in ('rot13','kazitext'):
            from training.sft.individual_fit.sft_fit import verify_fit_result
            output=self.artifacts/'training'/'sft'/stage.branch/stage.model/f'level_{stage.level:02d}'
            recorded=self.load(output/'fit_result.json')
            fit=verify_fit_result(output,recorded['identity'])
            if fit['identity']['configuration_hash'] != self.hash or fit['identity']['model_revision'] != model['revision'] or fit['identity']['tokenizer_revision'] != model['tokenizer_revision']:
                raise RuntimeError('Trained adapter does not match the current exact base/configuration')
            state.update(family=stage.branch,state_id=f'{stage.branch}_{stage.model}_level_{stage.level:02d}',level=stage.level,
                         adapter_path=fit['adapter_path'],checkpoint_hash=fit['checkpoint_hash'],adapter_manifest_hash=fit['adapter_manifest_hash'])
        elif stage.branch=='rl':
            trajectory=self.load(self.artifacts/'training'/'rl'/stage.model/'trajectory_report.json')
            model_name=model['id']
            run=next(r for r in trajectory['runs'] if r['model_id']==model_name)
            snapshot=next(s for s in run['snapshots'] if s['step']==stage.level)
            state.update(family='rl',state_id=f'rl_{stage.model}_step_{stage.level:03d}',step=stage.level,
                         adapter_path=snapshot['adapter_path'],checkpoint_hash=snapshot['checkpoint_hash'],adapter_manifest_hash=snapshot['artifact_hash'])

        return state

    def artifact_service(self):
        from operations.artifacts import ArtifactService
        if self._artifact_service is None:
            self._artifact_service=ArtifactService(Path(os.path.commonpath([self.data,self.artifacts])),self.artifacts/'state'/'artifact_registry.json',drive_store(self.config),drive_recovery=True)
        return self._artifact_service

    def prepared_collection_inputs(self, stage):
        """Reuse complete independently audited original CPU inputs explicitly."""
        input_root = os.environ.get('COLLECTION_INPUT_ROOT')
        certificate_path = os.environ.get('COLLECTION_INPUT_CERTIFICATE')
        if not input_root and not certificate_path:
            return None
        if not input_root or not certificate_path:
            raise RuntimeError('Prepared collection reuse requires both COLLECTION_INPUT_ROOT and COLLECTION_INPUT_CERTIFICATE')
        from operations.collection_inputs import validate_consumer_certificate
        from operations.drive import file_hash
        certificate = validate_consumer_certificate(certificate_path,self.config,consumer_commit=self.code,
            input_root=input_root,asset_cache=self.config.get('operations',{}).get('hf_cache') or self.cache)
        names = (('data/evaluation/capability_panel.json','data/evaluation/legibility_subset.json',
                  'data/evaluation/evaluation_package.json','data/evaluation/official_pools.json')
                 if stage.kind=='prepare_panels' else ('data/candidates/manifest.json',))
        inventory = {row['relative_path']:row for row in certificate['input_files']}
        service = self.artifact_service()
        outputs = []
        for name in names:
            info = inventory[name]
            source = Path(input_root)/name
            destination = self.data/Path(name).relative_to('data')
            with file_lock(str(destination)+'.copy.lock'):
                if destination.exists():
                    if destination.is_symlink() or destination.stat().st_size!=info['size_bytes'] or file_hash(destination)!=info['sha256']:
                        raise RuntimeError('Existing prepared collection input differs from the certified original')
                else:
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    descriptor,temporary_name = tempfile.mkstemp(prefix='.collection_input_',dir=destination.parent)
                    temporary = Path(temporary_name)
                    try:
                        with os.fdopen(descriptor,'wb') as output,source.open('rb') as original:
                            shutil.copyfileobj(original,output,length=8*1024*1024)
                            output.flush();os.fsync(output.fileno())
                        if temporary.stat().st_size!=info['size_bytes'] or file_hash(temporary)!=info['sha256']:
                            raise RuntimeError('Prepared collection copy differs from the certified original')
                        os.replace(temporary,destination);fsync_directory(destination.parent)
                    finally:
                        temporary.unlink(missing_ok=True)
            service.register(destination,{**self.provenance(stage),'producer':'collection_input_revalidator',
                'original_producer_commit':certificate['original_producer_commit'],
                'original_configuration_hash':certificate['original_configuration_hash'],
                'original_file_sha256':info['sha256'],'consumer_certificate_hash':certificate['certificate_hash']})
            outputs.append(str(destination))
        report = self.artifacts/'configurations'/f'{stage.kind}_input_reuse.json'
        atomic_write_json(report,{'status':'VERIFIED','stage':stage.kind,'consumer_commit':self.code,
            'consumer_configuration_hash':self.hash,'original_producer_commit':certificate['original_producer_commit'],
            'original_configuration_hash':certificate['original_configuration_hash'],
            'original_inventory_sha256':certificate['original_inventory_sha256'],
            'consumer_certificate_hash':certificate['certificate_hash'],
            'original_bytes_preserved':True,'input_files':[inventory[name] for name in names]})
        service.register(report,{**self.provenance(stage),'producer':'collection_input_revalidator',
            'original_producer_commit':certificate['original_producer_commit'],
            'consumer_certificate_hash':certificate['certificate_hash']})
        if service.sync_artifacts()['status']!='VERIFIED':
            raise RuntimeError('Validated prepared collection inputs have incomplete Drive replication')
        return {'status':'VERIFIED','output_files':outputs+[str(report)],
                'input_policy':'independently_validated_original_bytes','consumer_certificate_hash':certificate['certificate_hash']}

    def action(self, stage):
        kind=stage.kind
        if kind=='provision':
            from operations.assets import assets_from_configuration,provision_asset
            manifests=[]
            for asset in assets_from_configuration(self.config):
                if asset['repo_type']=='model': manifests.append(provision_asset(asset,self.cache,tokenizer_only=True))
            return self.write(self.artifacts/'configurations'/'public_asset_manifest.json',{'status':'tokenizers_verified','assets':manifests,'configuration_hash':self.hash})
        if kind=='prepare_panels':
            prepared=self.prepared_collection_inputs(stage)
            if prepared is not None:return prepared
            from data_generation_collection.evaluation_data.evaluation_data_preparation import prepare_evaluation_data
            pools=self.source_pools()
            sandbox=self.sandbox()
            result=prepare_evaluation_data(pools,{k:v['revision'] for k,v in self.config['datasets'].items() if k!='lmsys'},output_dir=self.data/'evaluation',verify_mbpp_reference=lambda code,tests:sandbox.execute(code,tests).get('correct') is True)
            if result['capability_panel']['status']!='verified' or result['legibility_subset']['status']!='verified':
                raise RuntimeError('Frozen panel/reference validation incomplete')
            return {'status':'VERIFIED','output_files':[str(p) for p in (self.data/'evaluation').glob('*.json')]}
        if kind=='prepare_candidates':
            prepared=self.prepared_collection_inputs(stage)
            if prepared is not None:return prepared
            from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import load_lmsys_source,prepare_candidate_manifest
            authorized=self.config.get('operations',{}).get('lmsys_authorization') is True
            revision=self.config['datasets']['lmsys']['revision']
            rows=load_lmsys_source(revision,authorized=authorized,cache_root=self.cache)
            questions=[r.get('question',r.get('prompt',r.get('text',''))) for rows_ in self.source_pools().values() for r in rows_]
            result=prepare_candidate_manifest(rows,self.tokenizers(),revision,questions,authorized=authorized,output_path=self.data/'candidates'/'manifest.json')
            return {'status':'VERIFIED','output_files':[str(self.data/'candidates'/'manifest.json')]}
        if kind=='collect_clean':
            from data_generation_collection.sft_data.clean_trace_collection.clean_trace_collection import collect_clean_traces
            from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import NativeCleanGenerator
            generator=NativeCleanGenerator(self.config['models'],cache_root=self.cache)
            backend=self.kazi()
            try:
                result=collect_clean_traces(self.load(self.data/'candidates'/'manifest.json'),generator,self.tokenizers(),kazitext=backend,output_dir=self.data/'clean_traces',config_hash=self.hash)
            finally:
                backend.close()
            if result.get('status')=='paused':
                return {'status':'PAUSED','resume':result,
                        'output_files':[str(p) for p in (self.data/'clean_traces').rglob('*')
                                        if p.is_file() and not p.name.endswith('.lock') and p.name!='STOP_REQUESTED']}
            published=self.write(self.data/'accepted'/'accepted_pool.json',result)
            published['output_files'] += [str(p) for p in (self.data/'clean_traces').rglob('*') if p.is_file() and not p.name.endswith('.lock')]
            return published
        if kind=='generation_preflight':
            from shared_procedures.preflight import run_preflights,require_preflights,preflight_requirements
            report=run_preflights(self.config,output_dir=self.artifacts/'preflight'/'generation',execute_live=True,scope='generation')
            require_preflights(report,preflight_requirements('generation'),expected_configuration_hash=self.hash)
            return {'status':'VERIFIED','output_files':[str(self.artifacts/'preflight'/'generation'/'report.json')]}
        if kind=='transform':
            from data_generation_collection.sft_data.sft_data_preparation import build_sft_datasets
            backend=self.kazi()
            try:
                result=build_sft_datasets(self.load(self.data/'accepted'/'accepted_pool.json'),self.tokenizers(),self.data/'transformed',config_hash=self.hash,kazitext=backend)
            finally:
                backend.close()
            published=self.write(self.data/'transformed'/'sft_manifest.json',result)
            published['output_files'] += [str(p) for p in (self.data/'transformed').rglob('*') if p.is_file() and str(p) not in published['output_files']]
            return published
        if kind=='prepare_rl':
            from data_generation_collection.rl_data.rl_data_preparation import prepare_rl_data,load_rl_source
            revision=self.config['datasets']['gsm8k']['revision']
            rows=load_rl_source(revision,cache_root=self.cache)
            questions=[r.get('question',r.get('prompt',r.get('text',''))) for rows_ in self.source_pools().values() for r in rows_]
            result=prepare_rl_data(rows,revision,questions,output_path=self.data/'rl'/'manifest.json')
            return {'status':'VERIFIED','output_files':[str(self.data/'rl'/'manifest.json')]}
        if kind=='freeze_analysis':
            from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan
            package=self.load(self.data/'evaluation'/'evaluation_package.json')
            plan=create_analysis_plan(package['capability_panel'],package['legibility_subset'],self.config)
            return self.write(self.data/'evaluation'/'analysis_plan.json',plan)
        if kind=='validate_statistics':
            from data_generation_collection.evaluation_data.protocol.analysis_plan.statistical_validation import run_statistical_validation
            plan=self.load(self.data/'evaluation'/'analysis_plan.json')
            report=run_statistical_validation(plan,output_path=self.artifacts/'preflight'/'statistical_validation.json')
            status='VERIFIED' if report.get('ready_for_confirmatory_launch') is True else 'FAILED'
            return self.write(self.artifacts/'preflight'/'statistical_validation.json',report,status)
        if kind in ('baseline','evaluate'):
            from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import NativeGenerator
            from evaluation.shared_model_evaluation.model_evaluation import evaluate_model
            protocol=self.protocol(trained=kind=='evaluate')
            generator=NativeGenerator(self.artifacts/'evaluation')
            states=self.state(stage)
            for seed in ((42,43,44) if kind=='baseline' else (42,)):
                result=evaluate_model(states,protocol,generator,self.judge() if seed==42 else None,sandbox=self.sandbox(),cache_root=self.artifacts/'evaluation',seed=seed)
                self.write(self.artifacts/'evaluation'/f'{states["state_id"]}_seed_{seed}.json',result)
                if result['status']!='complete': raise RuntimeError('Native evaluation incomplete; valid raw records retained')
            return {'status':'VERIFIED','output_files':[str(self.artifacts/'evaluation'/f'{states["state_id"]}_seed_{seed}.json') for seed in ((42,43,44) if kind=='baseline' else (42,))]}
        if kind=='repeatability':
            from data_generation_collection.evaluation_data.protocol.analysis_plan.baseline_repeatability import baseline_repeatability
            passes={m:{s:self.load(self.artifacts/'evaluation'/f'base_{m}_seed_{s}.json') for s in (42,43,44)} for m in ('7b','14b')}
            report=baseline_repeatability(passes,self.load(self.data/'evaluation'/'evaluation_package.json')['capability_panel'])
            return self.write(self.artifacts/'preflight'/'baseline_repeatability.json',report,'VERIFIED' if report.get('ready_for_training') is True else 'FAILED')
        if kind=='preflight':
            from shared_procedures.preflight import run_preflights
            evidence_path=self.artifacts/'preflight'/'live_checks.json'
            checks={}
            if evidence_path.exists():
                evidence=self.load(evidence_path)
                if evidence.get('configuration_hash')!=self.hash or evidence.get('report_hash')!=content_hash({k:v for k,v in evidence.items() if k!='report_hash'}):
                    raise RuntimeError('Live preflight evidence is stale or corrupt')
                checks={row['name']:row for row in evidence['checks']}
            for name in ('baseline_repeatability','statistical_validation'):
                report=self.load(self.artifacts/'preflight'/f'{name}.json')
                if report.get('report_hash')!=content_hash({k:v for k,v in report.items() if k!='report_hash'}):
                    raise RuntimeError(f'{name} report integrity failed')
                for row in report.get('checks',[]):
                    if row['name']==name:
                        checks[name]=row
            report=run_preflights(self.config,checks=checks,output_dir=self.artifacts/'preflight',execute_live=True)
            return {'status':'VERIFIED' if report['ready'] else 'BLOCKED','output_files':[str(self.artifacts/'preflight'/'report.json')]}
        if kind=='sft':
            from training.sft.sft_training import build_worklist,load_dataset_bundle
            from training.sft.individual_fit.sft_fit import run_sft_fit
            bundles=self.load(self.data/'transformed'/'sft_manifest.json')['datasets']
            task=next(t for t in build_worklist(self.config,bundles) if (t['family'],t['model_id'],t['level'])==(stage.branch,stage.model,stage.level))
            train,validation=load_dataset_bundle(task['dataset'])
            out=self.artifacts/'training'/'sft'/stage.branch/stage.model/f'level_{stage.level:02d}'
            result=run_sft_fit(task['spec'],train,validation,out,self.config,preflight_report=self.load(self.artifacts/'preflight'/'report.json'))
            published=self.write(out/'fit_result.json',result,'VERIFIED' if result['status']=='complete' else 'FAILED')
            published['output_files'] += [str(p) for p in (out/'final_adapter').rglob('*') if p.is_file()]
            published['output_files'] += [str(p) for p in (out/'checkpoint_001250').rglob('*') if p.is_file()]
            return published
        if kind=='rl':
            from training.rl_grpo.rl_training import run_rl_training
            revisions={v['id']:v['revision'] for v in self.config['models'].values()}
            tokenizer_revisions={v['id']:v['tokenizer_revision'] for v in self.config['models'].values()}
            output=self.artifacts/'training'/'rl'/stage.model
            report=run_rl_training(self.load(self.data/'rl'/'manifest.json'),output,judge=self.judge(),model_revisions=revisions,tokenizer_revisions=tokenizer_revisions,model_sizes=(stage.model,),runtime_identity={**self.config['runtime'],'configuration_hash':self.hash},preflight_evidence=self.load(self.artifacts/'preflight'/'report.json'))
            published=self.write(output/'trajectory_report.json',report,'VERIFIED' if report['status']=='completed' else 'FAILED')
            published['output_files'] += [str(p) for p in output.rglob('*') if p.is_file() and not p.name.endswith('.lock') and str(p) not in published['output_files']]
            return published
        if kind=='analyze':
            from evaluation.results_analysis.results_analysis import analyze_results
            records={p.stem.removesuffix('_seed_42'):self.load(p) for p in (self.artifacts/'evaluation').glob('*_seed_42.json')}
            report=analyze_results(records,self.load(self.data/'evaluation'/'analysis_plan.json'),validation_report=self.load(self.artifacts/'preflight'/'statistical_validation.json'),confirmatory=True)
            return self.write(self.artifacts/'reports'/'analysis.json',report)
        if kind=='report':
            from operations.reporting import generate_report
            files=generate_report(self.load(self.artifacts/'reports'/'analysis.json'),self.artifacts/'reports')
            return {'status':'VERIFIED','output_files':[str(p) for p in files]}
        if kind=='sync':
            service=self.artifact_service()
            # Only known stage outputs supply scientific provenance. Existing
            # registry records retain their original identities across versions.
            registered={r['local_path'] for r in service.list_artifacts()}
            for stage_id,node in self.runner.graph.items():
                status=self.runner.status(stage_id)
                if status['status']!='COMPLETED':
                    continue
                for path in status.get('result',{}).get('output_files',[]):
                    path=Path(path).resolve()
                    if str(path) not in registered and not path.is_relative_to(self.artifacts/'state'):
                        service.register(path,self.provenance(node))
                        registered.add(str(path))
            report=service.sync_artifacts(verify_existing=True)
            return self.write(self.artifacts/'state'/'synchronization_report.json',report,report['status'])
        raise ValueError(f'Unknown scientific stage kind: {kind}')

    def provenance(self, stage=None, metadata=None):
        stage=stage or self._current_stage
        metadata=metadata or {}
        identity=metadata.get('identity',{})
        if not identity and stage and stage.kind=='sft':
            path=self.artifacts/'training'/'sft'/stage.branch/stage.model/f'level_{stage.level:02d}'/'run_identity.json'
            identity=self.load(path) if path.exists() else {}
        model=(stage.model if stage else None) or identity.get('model_id') or 'all'
        model=next((key for key,value in self.config['models'].items() if value['id']==model),model)
        model_spec=self.config['models'].get(model,{})
        return {'experiment_id':self.hash,'branch':(stage.branch if stage else None) or identity.get('family') or 'shared',
            'model':model,'run_id':(stage.stage_id if stage else None) or identity.get('run_id') or self.hash[:16],
            'stage':stage.kind if stage else 'runtime','producer':'experiment_overview',
            'source_code_commit':self.code,'configuration_hash':self.hash,
            'dataset_revision':identity.get('dataset_hash') or identity.get('source_hash') or content_hash(self.config['datasets']),
            'model_revision':identity.get('model_revision') or model_spec.get('revision') or content_hash(self.config['models']),'seeds':[42]}

    def publish_permanent(self, path, metadata):
        path=Path(path).resolve()
        if not (path.is_relative_to(self.data) or path.is_relative_to(self.artifacts)):
            raise ValueError('Scientific artifact is outside configured data/artifact roots')
        if path.name.endswith('.lock') or path.is_relative_to(self.artifacts/'state'):
            return
        service=self.artifact_service()
        from operations.drive import file_hash
        if not service.list_artifacts(local_path=str(path),sha256=file_hash(path)):
            service.register(path,self.provenance(metadata=metadata))
        if metadata.get('defer_until_batch_end'):
            return
        result=service.sync_artifacts(workers=int(os.getenv('DRIVE_UPLOAD_WORKERS','1')))
        if result['status']!='VERIFIED':
            raise RuntimeError('Permanent artifact replication is incomplete; valid local state preserved')

    def _execute_stage(self,stage):
        self._current_stage=stage
        try:
            result=self.action(stage)
            for path in result.get('output_files',[]):
                notify_artifact(path,{'stage_id':stage.stage_id,'scientific_status':result.get('status')})
            self.publish_worker_state()
            return result
        finally:
            self._current_stage=None

    def publish_worker_state(self):
        """Replicate coherent operational ledgers at completed/pause boundaries."""
        if self._artifact_service is None:return
        from shared_procedures.recovery_procedure import RecoveryManager
        from shared_procedures.cost_accounting import CostLedger
        from uuid import uuid4
        worker=os.environ.get('ILLEGIBILITY_WORKER_ID','coordinator')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',worker):
            raise ValueError('Worker state snapshot requires a stable safe worker identity')
        recovery=RecoveryManager.for_worker(self.artifacts)
        destination=self.artifacts/'operations'/'worker_state'/worker/uuid4().hex
        files=[]
        if recovery.database_path.is_file():
            files.append(recovery.snapshot_database(destination/'retries.sqlite3'))
        costs=CostLedger(self.artifacts/'state'/'spend.json',authorization=self.config['spending'])
        if costs.database_path.is_file():
            files.append(costs.snapshot_database(destination/'spend.sqlite3'))
        for path in files:self.publish_permanent(path,{'kind':'operational_state_snapshot'})
        # Include the just-verified ledger copies in the restoration inventory.
        registry=self._artifact_service.export_manifest()
        atomic_write_json(destination/'artifact_registry.json',read_json(registry))
        self.publish_permanent(destination/'artifact_registry.json',{'kind':'operational_state_snapshot'})

    def execute(self,targets):
        stages=topological_order(self.runner.graph,list(targets))
        paid=any((self.runner.graph[t].gpu_count or self.runner.graph[t].paid_services)
                 and self.runner.status(t)['status']!='COMPLETED' for t in stages)
        if paid:
            from shared_procedures.recovery_procedure import configure_worker_recovery
            configure_worker_recovery(self.artifacts)
            validate_configuration(self.config,runtime=True)
            if not re.fullmatch(r'[0-9a-f]{40}',self.code) or self.config['runtime']['implementation_revision']!=self.code:
                raise RuntimeError('Frozen runtime implementation revision must match the executed image/source commit')
            if (ROOT/'.git').exists():
                dirty=subprocess.run(['git','status','--porcelain','--untracked-files=normal','--','src','scripts','prompts','schemas','requirements.lock','pyproject.toml','dockerfile','deploy'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
                if dirty:
                    raise RuntimeError('Commit scientific source/dependency changes before paid execution')
            spend=self.config['spending']
            if not spend.get('authorized') or not spend.get('authorization_evidence'):
                raise RuntimeError('Current explicit GPU/API spending authorization is required')
            freeze_configuration(self.config,self.artifacts/'configurations'/'resolved_config.json')
            service=self.artifact_service()
            service.drive.create_project_folder()
            # Recover any pending replication before allowing additional paid work.
            if service.sync_artifacts()['status']!='VERIFIED':
                raise RuntimeError('Existing artifact replication remains incomplete')
            if 'clean' in stages:
                expected={key:self.provenance()[key] for key in ('configuration_hash','source_code_commit','dataset_revision','model_revision','seeds','model')}
                service.assert_collection_reconciled(expected)
            with artifact_hook(self.publish_permanent):
                for stage_id in stages:
                    status=self.runner.status(stage_id)
                    if status['status']=='COMPLETED':
                        self._current_stage=self.runner.graph[stage_id]
                        for path in status.get('result',{}).get('output_files',[]):
                            notify_artifact(path,{'stage_id':stage_id})
                self._current_stage=None
                return self.runner.run(targets,{s.kind:self._execute_stage for s in self.runner.graph.values()})
        return self.runner.run(targets,{s.kind:self._execute_stage for s in self.runner.graph.values()})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'config.yaml'))
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('hardware','plan','status','provision','preflight','drive-init','recover-collection','sync','validate','smoke'):
        p=sub.add_parser(name)
        if name=='plan': p.add_argument('--target',action='append',default=None)
        if name=='provision': p.add_argument('--include-weights',action='store_true');p.add_argument('--offline',action='store_true')
        if name=='preflight': p.add_argument('--execute-live',action='store_true')
        if name=='preflight': p.add_argument('--scope',choices=('generation','training'),default='training')
    run=sub.add_parser('run');run.add_argument('--target',action='append',default=None)
    authorization=sub.add_parser('drive-authorize');authorization.add_argument('--port',type=int,default=8765)
    retrieve=sub.add_parser('retrieve');retrieve.add_argument('artifact_id');retrieve.add_argument('destination');retrieve.add_argument('--expected-provenance',required=True)
    deploy=sub.add_parser('runpod-deploy');deploy.add_argument('--payload',required=True);deploy.add_argument('--authorization',required=True)
    pod=sub.add_parser('runpod-status');pod.add_argument('--pod-id')
    args=parser.parse_args(argv)
    try:
        config=load_configuration(args.config)
        experiment=Experiment(config,config_path=args.config)
        config=experiment.config
        if args.command in ('run','provision','drive-init','drive-authorize','recover-collection','sync','retrieve','runpod-deploy') or (args.command=='preflight' and args.execute_live):
            from shared_procedures.recovery_procedure import configure_worker_recovery
            configure_worker_recovery(experiment.artifacts)
        if args.command=='hardware':
            from operations.hardware import inspect_hardware
            result=inspect_hardware(experiment.artifacts)
        elif args.command in ('plan','status'): result=experiment.runner.inspect((args.target if args.command=='plan' else None) or ['sync'])
        elif args.command=='run': result=experiment.execute(args.target or ['sync'])
        elif args.command=='provision':
            from operations.assets import assets_from_configuration,provision_asset
            result=[provision_asset(a,experiment.cache,offline=args.offline,tokenizer_only=(a['repo_type']=='model' and not args.include_weights),authorization=config.get('operations',{}).get('asset_authorizations',{})) for a in assets_from_configuration(config) if a['asset_id']!='dataset_lmsys']
        elif args.command=='preflight':
            from shared_procedures.preflight import run_preflights
            if args.execute_live:
                validate_configuration(config,runtime=True)
                if not config['spending'].get('authorized') or not config['spending'].get('authorization_evidence'):
                    raise RuntimeError('Live preflights require current explicit GPU/API spending authorization')
                if config['runtime']['implementation_revision']!=experiment.code:
                    raise RuntimeError('Live preflights require a matching pinned build/source revision')
                freeze_configuration(config,experiment.artifacts/'configurations'/'resolved_config.json')
                experiment.artifact_service().drive.create_project_folder()
                with artifact_hook(experiment.publish_permanent):
                    output=experiment.artifacts/'preflight'/'generation' if args.scope=='generation' else experiment.artifacts/'preflight'
                    result=run_preflights(config,output_dir=output,execute_live=True,scope=args.scope)
            else:
                output=experiment.artifacts/'preflight'/'generation' if args.scope=='generation' else experiment.artifacts/'preflight'
                result=run_preflights(config,output_dir=output,scope=args.scope)
        elif args.command=='drive-authorize':
            from operations.drive import authorize
            client=os.environ.get('GOOGLE_OAUTH_CLIENT_SECRET_FILE');token=os.environ.get('GOOGLE_OAUTH_TOKEN_FILE')
            if not client or not token: raise RuntimeError('GOOGLE_OAUTH_CLIENT_SECRET_FILE and GOOGLE_OAUTH_TOKEN_FILE are required')
            authorize(client,token,args.port); result={'status':'AUTHORIZED','next':'drive-init'}
        elif args.command=='drive-init':
            store=drive_store(config);folder=store.create_project_folder();result=store.roundtrip(folder,experiment.hash);result['folder_id']=folder
            atomic_write_json(experiment.artifacts/'state'/'drive_folder.json',{'folder_id':folder})
        elif args.command=='recover-collection':
            if config['runtime']['implementation_revision']!=experiment.code:
                raise RuntimeError('Collection recovery requires the same pinned source revision')
            expected={key:experiment.provenance()[key] for key in ('configuration_hash','source_code_commit','dataset_revision','model_revision','seeds','model')}
            result=experiment.artifact_service().restore_collection(expected,data_root=experiment.data)
        elif args.command=='sync': result=experiment.action(experiment.runner.graph['sync'])
        elif args.command=='retrieve':
            expected=read_json(args.expected_provenance)
            if expected.get('configuration_hash')!=experiment.hash:
                raise RuntimeError('Retrieval provenance differs from the current scientific configuration')
            result={'path':str(experiment.artifact_service().retrieve_checkpoint(args.artifact_id,args.destination,expected))}
        elif args.command.startswith('runpod'):
            from operations.runpod import RunPodClient
            client=RunPodClient()
            result=client.create_pod(read_json(args.payload),read_json(args.authorization)) if args.command=='runpod-deploy' else (client.inspect_pod(args.pod_id) if args.pod_id else client.list_pods())
        elif args.command=='validate':
            from operations.validation import validate_repository
            result=validate_repository(ROOT)
        elif args.command=='smoke':
            command=[sys.executable,'-m','pytest','tests/integration','tests/unit/operations','-q']
            return subprocess.run(command,cwd=ROOT).returncode
        print(json.dumps(result,indent=2,default=str))
        if isinstance(result,dict) and (result.get('status') in ('BLOCKED','FAILED') or result.get('ready') is False): return 2
        if isinstance(result,list) and any(isinstance(row,dict) and row.get('status')=='PAUSED' for row in result): return 3
        return 0
    except (ValueError,RuntimeError,OSError,ImportError,subprocess.SubprocessError) as error:
        print(json.dumps({'status':'BLOCKED','reason':str(error)}),file=sys.stderr)
        return 2

if __name__=='__main__': raise SystemExit(main())
