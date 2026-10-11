"""Launch gate evidence, keeping local unit checks separate from live validation."""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
from pathlib import Path
from typing import Callable,Sequence
import os
from .configuration import validate_configuration,configuration_hash,snapshot_hash,runtime_requirements,ConfigurationError
from .execution_state import timestamp
from .identity import content_hash
from .storage import atomic_write_json,read_json

NAMES=('configuration','model_runtime','transformation_runtime','clean_data','loss_mask_gradients','full_checkpoint_resume','durable_storage',
       'judge_services','mbpp_sandbox','grpo_update','spending_authorization','baseline_repeatability','statistical_validation')
LIVE_NAMES=set(NAMES)-{'configuration'}
PREFLIGHT_SCOPES={
    'generation':('configuration','model_runtime','transformation_runtime','durable_storage','spending_authorization'),
    'training':NAMES,
}
class PreflightBlocked(RuntimeError):
    pass
@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    evidence: dict=field(default_factory=dict)
    required: bool=True
    def __post_init__(self):
        if self.status not in {'passed','failed','blocked','skipped'}: raise ValueError('invalid preflight status')
        if not self.name: raise ValueError('preflight name required')
    def to_dict(self): return asdict(self)

def _configuration(config: dict) -> CheckResult:
    try:
        validate_configuration(config)
    except ConfigurationError as error:
        return CheckResult('configuration','failed',str(error))
    return CheckResult('configuration','passed','Fixed scientific settings validate; unresolved live identities remain gates.',
                       {'configuration_hash':configuration_hash(config),'runtime_unresolved':runtime_requirements(config),'unit_only':True})

def _model_runtime(config: dict) -> CheckResult:
    missing=runtime_requirements(config)
    try:
        import torch
        cuda=torch.cuda.is_available()
        hardware=[{'name':torch.cuda.get_device_name(i),'vram_bytes':torch.cuda.get_device_properties(i).total_memory}
                  for i in range(torch.cuda.device_count())]
    except ImportError:
        cuda=False;hardware=[]
    # CUDA availability alone is not a model/tokenizer/generation identity validation.
    return CheckResult('model_runtime','blocked','Actual pinned construction-model generation and native tokenizer preflight required.',
                       {'cuda_available':cuda,'hardware':hardware,'unresolved':missing,'live':False})

def _spending(config: dict) -> CheckResult:
    spend=config.get('spending',{})
    authorized=bool(spend.get('authorized') and spend.get('authorization_evidence') and spend.get('authorized_usd',0)>0)
    return CheckResult('spending_authorization','passed' if authorized else 'blocked',
                       'Current explicit spending authorization recorded.' if authorized else 'No current spending authorization or positive ceiling.',
                       {'live':authorized,'authorization_evidence':spend.get('authorization_evidence'),
                        'authorized_usd':spend.get('authorized_usd',0),'configuration_hash':configuration_hash(config)})

def _judge_services(config: dict) -> CheckResult:
    names=config.get('runtime',{}).get('credential_env',{})
    presence={provider:any(bool(os.getenv(name)) for name in variables) for provider,variables in names.items() if provider in {'openai','anthropic'}}
    return CheckResult('judge_services','blocked','Both real provider schema/version/quota/cache/retry/cost tests are required.',
                       {'credential_presence':presence,'live':False})

def _statistical_validation(config: dict) -> CheckResult:
    root=Path(config.get('runtime',{}).get('artifact_root','artifacts'))/'preflight'
    candidates=[root/'statistical_validation'/'report.json',root/'statistical_validation.json']
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:
        return CheckResult('statistical_validation','blocked','Prospective complete-family simulation validation is unavailable.',{'live':False})
    report=read_json(path)
    if report.get('report_hash')!=content_hash({k:v for k,v in report.items() if k!='report_hash'}):
        return CheckResult('statistical_validation','failed','Stored simulation report integrity failed.',{'live':False,'path':str(path)})
    try:
        from data_generation_collection.evaluation_data.protocol.analysis_plan.statistical_validation import implementation_hash
        current=implementation_hash()
    except ImportError:
        return CheckResult('statistical_validation','blocked','Analysis implementation unavailable to verify simulation provenance.',{'live':False})
    if report.get('implementation_hash')!=current:
        return CheckResult('statistical_validation','blocked','Analysis code changed after simulation; dependent evidence must be rerun.',{'live':False,'path':str(path)})
    rows=[r for r in report.get('checks',[]) if r.get('name')=='statistical_validation']
    if len(rows)!=1:
        return CheckResult('statistical_validation','failed','Stored simulation report lacks canonical gate.',{'live':False})
    row=rows[0]
    evidence={**row.get('evidence',{}),'report_hash':report['report_hash'],'path':str(path),
              'simulation_plan_hash':report.get('simulation_plan_hash'),'implementation_hash':current}
    return CheckResult('statistical_validation',row['status'],row['detail'],evidence)

def preflight_requirements(scope: str='training') -> tuple[str,...]:
    if scope not in PREFLIGHT_SCOPES:raise ValueError('preflight scope must be generation or training')
    return PREFLIGHT_SCOPES[scope]

def run_preflights(config: dict,checks: dict[str,Callable[[dict],CheckResult] | CheckResult] | None=None,
                   output_dir: str | Path | None=None,*,execute_live: bool=False,scope: str='training') -> dict:
    required_names=preflight_requirements(scope)
    defaults={'configuration':_configuration,'model_runtime':_model_runtime,'spending_authorization':_spending,
              'judge_services':_judge_services,'statistical_validation':_statistical_validation}
    overrides=checks or {}
    if execute_live:
        from .recovery_procedure import configure_worker_recovery
        configure_worker_recovery(config.get('runtime',{}).get('artifact_root','artifacts'))
        gpu_results={}
        def gpu_check(name):
            def run(_config):
                if not gpu_results:
                    try: gpu_results.update(_run_model_checks(_config,include_training=scope=='training'))
                    except Exception as error:
                        gpu_results.update({n:CheckResult(n,'failed',type(error).__name__+': '+str(error),{'live':False}) for n in ('model_runtime','loss_mask_gradients','full_checkpoint_resume')})
                return gpu_results[name]
            return run
        defaults.update({n:gpu_check(n) for n in ('model_runtime','loss_mask_gradients','full_checkpoint_resume')})
        defaults.update(judge_services=_run_judge_checks,durable_storage=_run_durable_checks,clean_data=_run_clean_checks)
        def transformation_check(_config):
            from data_generation_collection.sft_data.kazitext.kazitext_data_construction import run_transformation_preflight
            return run_transformation_preflight(_config,Path(_config['runtime'].get('artifact_root','artifacts'))/'preflight'/'transformation_runtime')
        defaults['transformation_runtime']=transformation_check
        def grpo_check(_config):
            from training.rl_grpo.setup.rl_setup import run_grpo_preflight
            return run_grpo_preflight(_config,Path(_config['runtime'].get('artifact_root','artifacts'))/'preflight'/'grpo_update')
        defaults['grpo_update']=grpo_check
        def mbpp_check(_config):
            from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.mbpp_sandbox import run_mbpp_preflight
            return run_mbpp_preflight(_config,Path(_config['runtime'].get('artifact_root','artifacts'))/'preflight'/'mbpp_sandbox')
        defaults['mbpp_sandbox']=mbpp_check
    results=[]
    for name in NAMES:
        if name not in required_names:
            results.append(CheckResult(name,'skipped',f'Outside the {scope} launch scope.',{'live':False,'out_of_scope':True},False).to_dict())
            continue
        check=overrides.get(name,defaults.get(name))
        if check is None:
            result=CheckResult(name,'blocked',f'Required live {name} evidence has not been supplied.',{'live':False})
        else:
            try: result=check(config) if callable(check) else check
            except Exception as error: result=CheckResult(name,'failed',f'{type(error).__name__}: {error}')
            if isinstance(result,dict): result=CheckResult(**result)
            if result.name!=name: raise ValueError('preflight returned another check identity')
        # A passed mocked check never supplies a mandatory live gate.
        if name in LIVE_NAMES and result.status=='passed' and (result.evidence.get('live') is not True or result.evidence.get('simulation') is True):
            result=CheckResult(name,'blocked','Only actual runtime evidence can satisfy this mandatory launch gate.',result.evidence,result.required)
        results.append(result.to_dict())
    ready=all(r['status']=='passed' for r in results if r['required'])
    report={'version':'preflight_evidence_v1','at':timestamp(),'configuration_hash':configuration_hash(config),'snapshot_hash':snapshot_hash(config),
            'scope':scope,'required_names':list(required_names),'checks':results,'ready':ready,
            'training_ready':scope=='training' and ready,'study_ready':scope=='training' and ready}
    report['report_hash']=content_hash(report)
    if output_dir is not None: atomic_write_json(Path(output_dir)/'report.json',report)
    return report

def require_preflights(report: dict,names: Sequence[str],*,expected_configuration_hash: str | None=None) -> None:
    if expected_configuration_hash is not None and report.get('configuration_hash')!=expected_configuration_hash:
        raise PreflightBlocked('preflight configuration identity differs')
    by_name={row['name']:row for row in report.get('checks',[])}
    bad=[]
    for name in names:
        row=by_name.get(name)
        if row is None or row.get('status')!='passed' or not row.get('evidence'):
            bad.append(name)
        elif name in LIVE_NAMES and (row['evidence'].get('live') is not True or row['evidence'].get('simulation') is True):
            bad.append(name)
    if bad: raise PreflightBlocked('mandatory launch gates unavailable: '+', '.join(bad))

def load_preflight_evidence(path: str | Path,config: dict) -> dict:
    report=read_json(path)
    if report.get('report_hash')!=content_hash({k:v for k,v in report.items() if k!='report_hash'}):
        raise PreflightBlocked('preflight report integrity invalid')
    if report.get('configuration_hash')!=configuration_hash(config):
        raise PreflightBlocked('stale preflight configuration')
    return report


def _run_model_checks(config: dict,*,include_training: bool=True) -> dict[str,CheckResult]:
    """Exercise both selected models on distinct nonexperimental native controls."""
    names=('model_runtime','loss_mask_gradients','full_checkpoint_resume')
    try:
        import torch
    except ImportError:
        return {n:CheckResult(n,'blocked','PyTorch runtime unavailable.',{'live':False}) for n in names}
    if not torch.cuda.is_available():
        return {n:CheckResult(n,'blocked','Actual selected-model preflight requires compatible CUDA hardware.',{'live':False}) for n in names}
    spending=config.get('spending',{})
    if not spending.get('authorized') or not spending.get('authorization_evidence'):
        return {n:CheckResult(n,'blocked','Current compute spending authorization required.',{'live':False}) for n in names}
    validate_configuration(config,runtime=True)
    if not include_training:
        from data_generation_collection.sft_data.clean_trace_collection.candidate_batch import clean_trace_batch
        from .response_parsing import parse_native_output
        from .identity import derive_seed,file_hash
        from .storage import notify_artifact
        import gc
        control_prompt='A notebook has three blank pages. One more blank page is added. How many blank pages are there now?'
        control_hash=content_hash({'kind':'nonexperimental_native_clean_control_v1','prompt':control_prompt})
        root=Path(config['runtime'].get('artifact_root','artifacts'))/'preflight'/'model_runtime'
        generator=clean_trace_batch.NativeCleanGenerator(config['models'],device_map='auto',torch_dtype='bfloat16',
                                                         cache_root=config['runtime'].get('cache_root'))
        records=[]
        try:
            for size,model_spec in config['models'].items():
                candidate={'example_id':'control_native_clean_'+size+'_'+control_hash[:16],
                           'prompt':control_prompt,'source_content_hash':content_hash(control_prompt)}
                responses=list(generator.generate_many(size,[candidate]))
                if len(responses)!=1:
                    raise PreflightBlocked('Native clean control must return exactly one response per selected model')
                response=responses[0]
                model=generator.model
                dtypes=sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()})
                quantized=bool(getattr(model,'is_loaded_in_4bit',False) or getattr(model,'is_loaded_in_8bit',False)
                               or getattr(model.config,'quantization_config',None))
                adapter=bool(getattr(model,'peft_config',None) or getattr(model,'_hf_peft_config_loaded',False))
                if dtypes!=[str(torch.bfloat16)] or quantized or adapter:
                    raise PreflightBlocked('Native clean control did not load the untouched unquantized BF16 base')
                ids=response.get('generation_token_ids',[])
                eos=model.generation_config.eos_token_id
                eos_ids=eos if isinstance(eos,list) else [eos]
                prefix=response.get('native_reasoning_prefix','')
                raw=response.get('raw_generation')
                if (response.get('simulation') is not False or response.get('model_name')!=model_spec['id']
                    or response.get('model_revision')!=model_spec['revision']
                    or response.get('tokenizer_revision')!=model_spec['tokenizer_revision']
                    or response.get('seed')!=derive_seed(42,'clean_generation',candidate['example_id'],size)
                    or response.get('generation_settings')!=clean_trace_batch.GENERATION_SETTINGS
                    or prefix not in ('','<think>') or not isinstance(raw,str)
                    or response.get('generated_token_count')!=len(ids) or not ids or ids[-1] not in eos_ids
                    or response.get('finish_reason')!='stop'):
                    raise PreflightBlocked('Native clean control generation identity, decoding policy or EOS completion differs')
                parsed=parse_native_output(prefix+raw)
                runtime={**response.get('runtime',{}),'loader':'NativeCleanGenerator','base_dtype':str(model.dtype),
                    'floating_parameter_dtypes':dtypes,'quantization':None,'adapter_loaded':False,'optimizer_loaded':False,
                    'device_map':{key:str(value) for key,value in getattr(model,'hf_device_map',{}).items()},
                    'model_device':str(model.device),'generator_implementation_hash':file_hash(clean_trace_batch.__file__)}
                raw_record={**response,'execution_mode':'control_preflight','control_preflight':True,
                    'scientific_result':False,'model':model_spec,'control_hash':control_hash,
                    'prompt':control_prompt,'example_id':candidate['example_id'],'raw_hash':content_hash(raw),
                    'runtime':runtime,'configuration_hash':configuration_hash(config),'parser_status':parsed['status'],
                    'eos_token_ids':eos_ids,'eos_complete':True}
                raw_path=root/('native_clean_control_'+size+'.json')
                atomic_write_json(raw_path,raw_record);notify_artifact(raw_path,raw_record)
                if parsed['status']!='valid':
                    raise PreflightBlocked('Actual native clean control did not yield a complete separated reasoning/final response')
                records.append({'model':model_spec,'raw_path':str(raw_path),'raw_hash':raw_record['raw_hash'],
                    'runtime':runtime,'generation_settings':response['generation_settings'],'seed':response['seed'],
                    'native_reasoning_prefix':prefix,'generated_token_count':len(ids),'eos_token_ids':eos_ids,
                    'eos_complete':True,'parser_status':parsed['status'],'control_hash':control_hash})
                if generator.batch_size_for(size) > 1:
                    companions = [candidate, {**candidate, 'example_id':candidate['example_id']+'_companion',
                        'prompt':'A notebook has three blank pages. After one additional blank page is inserted, how many blank pages does it contain?'}]
                    batched = list(generator.generate_many(size, companions))
                    if len(batched) != 2:
                        raise PreflightBlocked('Actual batched native control did not return both whole prompts')
                    controls = []
                    for index, (item, response_) in enumerate(zip(companions, batched)):
                        ids_ = response_.get('generation_token_ids', [])
                        parsed_ = parse_native_output(response_.get('native_reasoning_prefix','')+response_['raw_generation'])
                        if (response_.get('seed')!=derive_seed(42,'clean_generation',item['example_id'],size)
                            or response_.get('generation_settings')!=clean_trace_batch.GENERATION_SETTINGS
                            or response_.get('runtime',{}).get('generation_batch_size')!=2
                            or response_.get('finish_reason')!='stop' or not ids_ or ids_[-1] not in eos_ids
                            or parsed_['status']!='valid'):
                            raise PreflightBlocked('Actual batched native control identity, EOS or parser differs')
                        path_=root/f'native_clean_batch_control_{size}_{index}.json'
                        value_={**response_,'example_id':item['example_id'],'prompt':item['prompt'],
                            'execution_mode':'control_preflight','control_preflight':True,'scientific_result':False,
                            'configuration_hash':configuration_hash(config),'runtime':{**runtime,**response_['runtime']},
                            'raw_hash':content_hash(response_['raw_generation']),'eos_complete':True,'parser_status':parsed_['status']}
                        atomic_write_json(path_,value_);notify_artifact(path_,value_)
                        controls.append({'path':str(path_),'raw_hash':value_['raw_hash'],'seed':response_['seed'],
                                         'eos_complete':True,'parser_status':parsed_['status']})
                    records[-1]['batch_controls']=controls
                model=None
        finally:
            generator.model=generator.tokenizer=None
            generator=model=None
            gc.collect();torch.cuda.empty_cache()
        return {'model_runtime':CheckResult('model_runtime','passed','Both pinned untouched BF16 bases passed actual native clean controls.',
                    {'live':True,'simulation':False,'control_preflight':True,'scientific_results':False,
                     'runtime_kind':'native_clean_base_bfloat16','training_runtime_loaded':False,'models':records}),
                **{n:CheckResult(n,'skipped','Training microstate control is outside generation scope.',
                    {'live':False,'out_of_scope':True},False) for n in ('loss_mask_gradients','full_checkpoint_resume')}}
    from training.sft.individual_fit.sft_fit import (SFTFitSpec,_runtime,tokenize_completion,verify_mask_gradients,
        collate_completion,completion_loss,capture_rng,restore_rng)
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from .checkpointing import save_checkpoint,load_checkpoint,verify_checkpoint,_same_state
    from .response_parsing import parse_native_output
    from .identity import derive_seed
    from .storage import notify_artifact
    import gc
    control_prompt='What is two plus two?'
    control_completion='<think>Two objects combined with two objects make four objects.</think>\n4'
    control_hash=content_hash({'kind':'nonexperimental_live_runtime_control_v1','prompt':control_prompt,'completion':control_completion})
    root=Path(config['runtime'].get('artifact_root','artifacts'))/'preflight'/'model_runtime'
    evidence={n:[] for n in names}
    for size,model_spec in config['models'].items():
        model=tokenizer=optimizer=scheduler=None
        try:
            spec=SFTFitSpec('rot13',size,1,model_spec['revision'],model_spec['tokenizer_revision'],configuration_hash(config),control_hash)
            model,tokenizer,optimizer,scheduler,runtime=_runtime(spec,config['runtime'].get('sft_device','cuda:0'),cache_root=config['runtime'].get('cache_root'))
            model.eval()
            prompt=tokenizer.apply_chat_template([{'role':'user','content':control_prompt}],tokenize=False,add_generation_prompt=True)
            encoded=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(model.device)
            from transformers import set_seed
            set_seed(derive_seed(42,'control_preflight',size))
            with torch.inference_mode():
                generated=model.generate(**encoded,do_sample=True,temperature=.6,top_p=.95,top_k=0,repetition_penalty=1.,max_new_tokens=3072,pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            ids=generated[0,encoded['input_ids'].shape[1]:].tolist()
            raw=tokenizer.decode(ids,skip_special_tokens=True)
            prefix='<think>' if prompt.rstrip().endswith('<think>') and not raw.lstrip().startswith('<think>') else ''
            raw_record={'control_preflight':True,'scientific_result':False,'simulation':False,'model':model_spec,
                        'raw_output':raw,'raw_hash':content_hash(raw),'generated_tokens':len(ids),'runtime':runtime,
                        'configuration_hash':configuration_hash(config)}
            raw_path=root/('control_'+size+'.json');atomic_write_json(raw_path,raw_record);notify_artifact(raw_path,raw_record)
            parsed=parse_native_output(prefix+raw)
            eos=model.generation_config.eos_token_id
            eos_ids=eos if isinstance(eos,list) else [eos]
            if parsed['status']!='valid' or not ids or ids[-1] not in eos_ids:
                raise PreflightBlocked('Actual selected-model native control did not yield a complete separated reasoning/final response')
            evidence['model_runtime'].append({'model':model_spec,'raw_path':str(raw_path),'raw_hash':raw_record['raw_hash'],'runtime':runtime})
            example=tokenize_completion({'example_id':'control_'+size,'prompt':control_prompt,'completion':control_completion},tokenizer)
            pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            gradients=verify_mask_gradients(model,example,pad,require_lora=True)
            evidence['loss_mask_gradients'].append({'model':size,'checks':gradients,'control_hash':control_hash})
            model.train();batch=collate_completion([example],pad,device=model.device)
            optimizer.zero_grad(set_to_none=True)
            output=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
            loss=completion_loss(output.logits,batch['labels']);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.3)
            optimizer.step();scheduler.step()
            state={'adapter':get_peft_model_state_dict(model,save_embedding_layers=False),'optimizer':optimizer.state_dict(),
                   'scheduler':scheduler.state_dict(),'rng':capture_rng(),'progress':{'completed_steps':1,'sampler_position':1},'runtime':runtime}
            identity={**spec.identity(),'execution_mode':'control_preflight','control_hash':control_hash,'runtime_hash':content_hash(runtime)}
            directory=root/size/('checkpoint_'+content_hash(identity)[:16])
            manifest=save_checkpoint(directory,identity,state)
            restored=load_checkpoint(directory,identity)
            # Deliberately perturb live microstate; comparing an untouched object is not a resume test.
            with torch.no_grad():
                first=next(parameter for parameter in model.parameters() if parameter.requires_grad)
                first.add_(.25)
                for item in optimizer.state.values():
                    tensor=next((v for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point()),None)
                    if tensor is not None:
                        tensor.add_(.25)
                        break
            scheduler.step();torch.rand(1,device=model.device)
            if _same_state(get_peft_model_state_dict(model,save_embedding_layers=False),restored['adapter']):
                raise PreflightBlocked('Full-state control perturbation did not change adapter microstate')
            set_peft_model_state_dict(model,restored['adapter']);optimizer.load_state_dict(restored['optimizer'])
            scheduler.load_state_dict(restored['scheduler']);restore_rng(restored['rng'])
            if not _same_state(get_peft_model_state_dict(model,save_embedding_layers=False),restored['adapter']) or not _same_state(optimizer.state_dict(),restored['optimizer']) or not _same_state(scheduler.state_dict(),restored['scheduler']) or not _same_state(capture_rng(),restored['rng']):
                raise PreflightBlocked('Actual full optimizer state failed restoration equality')
            verify_checkpoint(directory,identity)
            evidence['full_checkpoint_resume'].append({'model':size,'checkpoint_hash':manifest['checkpoint_hash'],'original_step':1,'optimizer_restored':True})
        finally:
            model=tokenizer=optimizer=scheduler=None
            gc.collect();torch.cuda.empty_cache()
    return {n:CheckResult(n,'passed' if n=='model_runtime' or include_training else 'skipped','Both actual selected-model control preflights passed.' if n=='model_runtime' or include_training else 'Training microstate control is outside generation scope.',
             {'live':True,'simulation':False,'control_preflight':True,'scientific_results':False,'models':evidence[n]}) for n in names}


def _run_judge_checks(config: dict) -> CheckResult:
    from .judge_client import JudgeClient
    from .cost_accounting import CostLedger
    root=Path(config['runtime'].get('artifact_root','artifacts'))
    spend=config['spending'];judges=config['judges']
    if not spend.get('authorized') or not spend.get('authorization_evidence'):
        return CheckResult('judge_services','blocked','Current API spending authorization required.',{'live':False})
    client=JudgeClient(root/'judge_cache',root/'state',cost_ledger=CostLedger(root/'state'/'spend.json',authorization=spend),
        provider_prices=spend.get('provider_prices'),pricing_evidence=spend.get('pricing_evidence'),
        quota_limits=judges.get('quota_limits'),quota_evidence=judges.get('quota_evidence'))
    prompt_root=Path(__file__).resolve().parents[2]/'prompts'
    trace='Two quantities of two are combined. The sum follows by adding the quantities together.'
    question='What is two plus two?'
    requests=[]
    for kind,model,prompt_name,aware in [
        ('rl_pass_a',judges['reward_model'],'rl_pass_a',False),('rl_pass_b',judges['reward_model'],'rl_pass_b',True),
        ('evaluation_primary',judges['primary_model'],'evaluation_legibility',False),
        ('evaluation_secondary',judges['secondary_model'],'evaluation_legibility',False),
        ('evaluation_audit',judges['primary_model'],'evaluation_readable_solution',True)]:
        record=client.request(kind,model,(prompt_root/(prompt_name+'.txt')).read_text(),trace,question=question if aware else None)
        requests.append(record)
        if record['status']!='success' or record.get('simulation') is not False:
            return CheckResult('judge_services','blocked',f'Actual {kind} request unavailable: {record["status"]}',
                               {'live':False,'requests':[{'kind':r['request_kind'],'request_hash':r['request_hash'],'status':r['status']} for r in requests]})
        cached=client.request(kind,model,(prompt_root/(prompt_name+'.txt')).read_text(),trace,question=question if aware else None)
        if cached!=record: return CheckResult('judge_services','failed','Provider cache readback differs.',{'live':True})
    return CheckResult('judge_services','passed','All five actual provider schema/version/cache/cost checks passed.',
                       {'live':True,'simulation':False,'control_preflight':True,'scientific_results':False,
                        'requests':[{'kind':r['request_kind'],'request_hash':r['request_hash'],'returned_model':r['returned_model'],
                                     'provider_version':r.get('provider_version'),'request_profile_hash':client._provider_profile(r),
                                     'usage':r['usage'],'cost_usd':r.get('cost_usd')} for r in requests],
                        'version_guard_scope':'request_profile_v2'})


def _run_durable_checks(config: dict) -> CheckResult:
    durability=config.get('runtime',{}).get('durability_evidence')
    mode=durability.get('mode','runpod_lifecycle') if isinstance(durability,dict) else 'runpod_lifecycle'
    if mode=='google_drive_primary':
        return _run_drive_primary_checks(config,durability)
    if mode!='runpod_lifecycle':
        return CheckResult('durable_storage','failed','Unsupported durable-storage mode.',{'live':False})
    from .storage import publish_artifact,verify_artifact,notify_artifact
    root=Path(config['runtime'].get('artifact_root','artifacts'))/'preflight'/'durable_storage'
    root.mkdir(parents=True,exist_ok=True)
    identity={'configuration_hash':configuration_hash(config),'purpose':'durability_probe_v1'}
    path=root/'probe.json'
    record=publish_artifact(path,{'payload':'durable_probe_v1'},identity=identity)
    verify_artifact(path,expected_identity=identity)
    notify_artifact(path,record)
    # A flush or Drive upload alone cannot prove RunPod stop/delete survival.
    lifecycle=config.get('runtime',{}).get('durability_evidence')
    if not isinstance(lifecycle,dict) or lifecycle.get('live') is not True or not lifecycle.get('stop_survival_verified') or not lifecycle.get('independent_backup_verified'):
        return CheckResult('durable_storage','blocked','Local write/flush/readback completed; actual lifecycle survival and independent backup evidence required.',
                           {'live':False,'local_flush_verified':True,'probe_path':str(path),'probe_hash':record['content_hash']})
    if lifecycle.get('probe_hash')!=record['content_hash']:
        return CheckResult('durable_storage','failed','Lifecycle evidence names another persistence probe.',{'live':False})
    return CheckResult('durable_storage','passed','Actual recorded lifecycle survival and independent backup match the verified probe.',
                       {'live':True,'probe_path':str(path),'probe_hash':record['content_hash'],'lifecycle':lifecycle})


def _run_drive_primary_checks(config: dict,durability: dict) -> CheckResult:
    """Check a retained Drive copy and a fresh restore on every scoped launch."""
    import tempfile
    from operations.drive import DriveStore,drive_service,file_hash
    from .storage import verify_artifact
    record=durability.get('registered_probe')
    independent=durability.get('independent_restore')
    config_hash=configuration_hash(config)
    expected_identity={'configuration_hash':config_hash,'purpose':'durability_probe_v1'}
    expected_content=content_hash({'payload':'durable_probe_v1'})
    expected_provenance={'configuration_hash':config_hash,
        'source_code_commit':config['runtime'].get('implementation_revision'),
        'dataset_revision':content_hash(config['datasets']),'model_revision':content_hash(config['models']),
        'seeds':[config['seed']],'model':'all'}
    unavailable={'live':False,'mode':'google_drive_primary'}
    if (durability.get('live') is not True or durability.get('simulation') is True
        or not isinstance(record,dict) or not isinstance(independent,dict)
        or independent.get('simulation') is True):
        return CheckResult('durable_storage','blocked','Actual retained probe and independent Drive restore evidence required.',unavailable)
    if (any(record.get(key)!=value for key,value in expected_provenance.items())
        or independent.get('status')!='INDEPENDENT_DRIVE_RETRIEVAL_VERIFIED'
        or independent.get('pod_filesystem_read_for_retrieval') is not False
        or independent.get('source_commit')!=expected_provenance['source_code_commit']
        or independent.get('configuration_hash')!=config_hash
        or independent.get('probe_identity')!=expected_identity
        or independent.get('content_hash')!=expected_content
        or not isinstance(independent.get('observed_at'),str) or not independent['observed_at']
        or any(independent.get(key)!=record.get(key) for key in ('artifact_id','drive_file_id','sha256','md5','size'))):
        return CheckResult('durable_storage','failed','Independent Drive probe identity, complete-file digests or source provenance differ.',unavailable)
    token=os.environ.get('GOOGLE_OAUTH_TOKEN_FILE')
    folder=os.environ.get('GOOGLE_DRIVE_FOLDER_ID') or config.get('operations',{}).get('drive_folder_id')
    if not token or not folder:
        return CheckResult('durable_storage','blocked','Protected Google OAuth token and canonical project folder are required.',unavailable)
    store=None
    try:
        store=DriveStore(drive_service(token),folder,
            state_root=Path(config['runtime'].get('artifact_root','artifacts'))/'state'/'drive_durability')
        store.create_project_folder()
        # A distinct absent destination prevents DriveStore's ordinary local-copy
        # optimization from supplying apparent cloud recovery evidence.
        with tempfile.TemporaryDirectory(prefix='illegibility_drive_restore_') as scratch:
            restored=store.restore_registered_artifact(record,Path(scratch)/'probe.json',
                expected_provenance=expected_provenance)
            probe=verify_artifact(restored,expected_identity=expected_identity)
            if probe['content_hash']!=expected_content or probe['data']!={'payload':'durable_probe_v1'}:
                raise ValueError('Retained Drive persistence probe differs')
            full_sha256=file_hash(restored)
            full_md5=file_hash(restored,'md5')
            size=restored.stat().st_size
    except Exception as error:
        # Transport exceptions can include credentials; only the type is safe.
        return CheckResult('durable_storage','blocked','Actual Drive restore or integrity verification failed; paid work remains blocked.',
            {**unavailable,'error_type':type(error).__name__})
    finally:
        service=getattr(store,'service',None)
        if service is not None and hasattr(service,'close'):
            try:service.close()
            except Exception:pass
    return CheckResult('durable_storage','passed','Actual registered Drive metadata and a fresh complete-file restore verify; RunPod storage is scratch.',
        {'live':True,'simulation':False,'mode':'google_drive_primary','configuration_hash':config_hash,
         'source_code_commit':expected_provenance['source_code_commit'],'artifact_id':record['artifact_id'],
         'drive_file_id':record['drive_file_id'],'drive_folder_id':record['drive_folder_id'],
         'probe_identity':expected_identity,'probe_hash':probe['content_hash'],'sha256':full_sha256,'md5':full_md5,'size':size,
         'independent_restore':independent,'verification':'current_remote_metadata_and_fresh_drive_download',
         'runpod_storage_required':False})


def _run_clean_checks(config: dict) -> CheckResult:
    from data_generation_collection.sft_data.sft_data_preparation import validate_accepted_pool,validate_sft_bundle
    from data_generation_collection.sft_data.kazitext.kazitext_data_construction import OfficialKaziText
    root=Path(config['runtime'].get('data_root','data'))
    pool_path=root/'accepted'/'accepted_pool.json';bundle_path=root/'transformed'/'manifest.json'
    # Current preparation writes manifest.json; retain the older bundle location.
    if not bundle_path.is_file():bundle_path=root/'transformed'/'sft_manifest.json'
    if not pool_path.is_file() or not bundle_path.is_file():
        return CheckResult('clean_data','blocked','Actual frozen joint corpus and all forty SFT datasets are required.',{'live':False})
    pool=validate_accepted_pool(read_json(pool_path),configuration_hash(config))
    if any(trace.get('simulation') is not False for pair in pool['pairs'] for trace in pair['traces'].values()):
        return CheckResult('clean_data','failed','Fixture or undeclared generation mode cannot satisfy actual clean-corpus gate.',{'live':False})
    options=config.get('operations',{})
    udpipe=options.get('udpipe_path');dictionaries=options.get('aspell_dictionary_files')
    if not udpipe or not dictionaries:
        return CheckResult('clean_data','blocked','Pinned loadable UDPipe and English Aspell dictionary provenance required.',{'live':False})
    from transformers import AutoTokenizer
    cache=Path(config['runtime'].get('cache_root','/workspace/model_cache'))
    source=cache/'kazitext'
    if not source.is_dir():
        return CheckResult('clean_data','blocked','Pinned official KaziText source unavailable.',{'live':False})
    kazi=OfficialKaziText(source,udpipe,aspell_dictionary_files=dictionaries,
                         expected_udpipe_sha256=config['interventions']['kazitext']['udpipe_sha256'])
    try:
        actual_dictionary={'files':kazi.provenance['aspell_dictionaries'],'settings':kazi.provenance['aspell_settings']}
        if actual_dictionary!=config['interventions']['kazitext'].get('aspell_dictionary_provenance'):
            return CheckResult('clean_data','failed','Actual Aspell dictionary contents or settings differ from the scientific freeze.',{'live':True,'simulation':False})
        from operations.assets import model_load_kwargs
        tokenizers={size:AutoTokenizer.from_pretrained(spec['id'],revision=spec['tokenizer_revision'],
                    **model_load_kwargs(spec['id'],spec['tokenizer_revision'],cache,tokenizer_only=True),trust_remote_code=False)
                    for size,spec in config['models'].items()}
        result=validate_sft_bundle(read_json(bundle_path),pool,tokenizers=tokenizers,kazitext=kazi,recompute=True)
    finally:
        if hasattr(kazi,'close'):kazi.close()
    return CheckResult('clean_data','passed','All forty actual datasets and every source transformation/length/protection check recomputed.',
                       {'live':True,'simulation':False,'source_pool_hash':pool['content_hash'],'validation':result})
