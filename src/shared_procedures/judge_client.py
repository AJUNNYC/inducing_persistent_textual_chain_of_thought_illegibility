"""Strict provider transport with independent durable request/pass caches."""
from __future__ import annotations
from pathlib import Path
import json
import os
import threading
import time
from typing import Any
import httpx
import jsonschema
from .cost_accounting import CostLedger,SpendingBlocked,estimate_judge_cost
from .execution_state import timestamp
from .identity import content_hash
from .recovery_procedure import RecoveryManager,RetryableFailure,RecoveryStopped
from .storage import atomic_write_json,read_json,file_lock,notify_artifact

KINDS={'rl_pass_a','rl_pass_b','evaluation_primary','evaluation_secondary','evaluation_audit'}
SCORE_SCHEMA={'type':'object','properties':{'score':{'type':'integer','minimum':1,'maximum':9}},'required':['score'],'additionalProperties':False}
RL_PASS_B_SCHEMA={'type':'object','properties':{'readable_solution_present':{'type':'boolean'},'obvious_padding_present':{'type':'boolean'}},'required':['readable_solution_present','obvious_padding_present'],'additionalProperties':False}
AUDIT_SCHEMA={'type':'object','properties':{k:{'type':'string','enum':['present','not_detected','uncertain']} for k in ('readable_solution','padding')},'required':['readable_solution','padding'],'additionalProperties':False}

class JudgeError(RuntimeError):
    def __init__(self,message: str,*,status: str='transport_error',retryable: bool=False,retry_after: float | None=None):
        super().__init__(message)
        self.status=status
        self.retryable=retryable
        self.retry_after=retry_after

def _strict_json(raw: str) -> dict:
    def pairs(items):
        value={}
        for key,item in items:
            if key in value: raise ValueError('duplicate response field')
            value[key]=item
        return value
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite response')))

def default_settings(model: str) -> dict:
    if model=='gpt-4o-2024-11-20':
        return {'temperature':0,'max_tokens':64,'timeout_seconds':60}
    if model=='claude-sonnet-5-5':
        return {'max_tokens':64,'thinking':{'type':'between_tools'},'output_config':{'effort':'low'},'timeout_seconds':60}
    raise JudgeError('unapproved judge model',status='model_mismatch')

def _persist_record(path: Path,record: dict) -> None:
    record.pop('record_hash',None)
    record['record_hash']=content_hash(record)
    atomic_write_json(path,record)

class ProviderTransport:
    """SDK retries disabled by using a single verified HTTPS request per invocation."""
    def __call__(self,request: dict) -> dict:
        model=request['model'];settings=request['settings'];schema=request['schema']
        message=json.dumps({'reasoning':request['trace'],**({'question':request['question']} if request['question'] is not None else {})},ensure_ascii=False)
        timeout=settings.get('timeout_seconds',60)
        if model.startswith('gpt-'):
            token=os.getenv('MODEL_OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
            if not token: raise JudgeError('OpenAI credential unavailable',status='blocked')
            endpoint='https://api.openai.com/v1/chat/completions'
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}
            body={'model':model,'messages':[{'role':'system','content':request['prompt']},{'role':'user','content':message}],
                  'temperature':0,'max_tokens':64,'response_format':{'type':'json_schema','json_schema':{'name':request['kind'],'strict':True,'schema':schema}}}
        else:
            token=os.getenv('ANTHROPIC_API_KEY')
            if not token: raise JudgeError('Anthropic credential unavailable',status='blocked')
            endpoint='https://api.anthropic.com/v1/messages'
            headers={'x-api-key':token,'anthropic-version':'2023-06-01','Content-Type':'application/json'}
            # Anthropic structured output accepts enum but rejects numerical
            # minimum/maximum. Encode the identical finite score domain on wire;
            # retain the original schema for request identity and local validation.
            wire_schema = ({**schema, 'properties': {'score': {'type':'integer','enum':list(range(1,10))}}}
                           if schema == SCORE_SCHEMA else schema)
            body={'model':model,'system':request['prompt'],'messages':[{'role':'user','content':message}],
                  'max_tokens':64,'thinking':{'type':'between_tools'},'output_config':{'effort':'low','format':{'type':'json_schema','schema':wire_schema}}}
        try:
            response=httpx.post(endpoint,headers=headers,json=body,timeout=timeout)
        except httpx.TimeoutException as error:
            raise JudgeError('provider timeout; billing uncertain',status='timeout',retryable=True) from error
        except httpx.RequestError as error:
            raise JudgeError('provider connection unavailable',retryable=True) from error
        if response.status_code>=400:
            try: code=str(response.json().get('error',{}).get('type',''))
            except ValueError: code=''
            if code in {'insufficient_quota','billing_error','credit_balance_too_low','authentication_error','permission_error'} or response.status_code in {401,403,402}:
                raise JudgeError('provider credentials/quota/billing rejected',status='quota_error')
            retry_after=response.headers.get('retry-after')
            try: wait=float(retry_after) if retry_after else None
            except ValueError: wait=None
            raise JudgeError(f'provider HTTP status {response.status_code}',retryable=response.status_code==429 or response.status_code>=500,retry_after=wait)
        value=response.json()
        if model.startswith('gpt-'):
            choice=value['choices'][0]
            if choice['message'].get('refusal'):
                raise JudgeError('provider refusal',status='refusal')
            text=choice['message'].get('content') or ''
            usage={'input_tokens':value['usage']['prompt_tokens'],'output_tokens':value['usage']['completion_tokens']}
            finish=choice.get('finish_reason')
        else:
            text=''.join(block.get('text','') for block in value.get('content',[]) if block.get('type')=='text')
            usage={'input_tokens':value['usage']['input_tokens'],'output_tokens':value['usage']['output_tokens']}
            finish=value.get('stop_reason')
        return {'text':text,'model':value.get('model'),'usage':usage,'request_id':value.get('id'),
                'finish_reason':finish,'provider_response':value,'provider_version':value.get('system_fingerprint')}

class JudgeClient:
    def __init__(self,cache_root: str | Path,state_root: str | Path,*,transport=None,cost_ledger: CostLedger | None=None,
                 concurrency: int=8,provider_prices: dict | None=None,pricing_evidence: dict | None=None,quota_limits: dict | None=None,quota_evidence: dict | None=None):
        if type(concurrency) is not int or not 1<=concurrency<=8:
            raise ValueError('judge concurrency must respect <=8 and actual account limits')
        self.cache_root=Path(cache_root);self.state_root=Path(state_root)
        self.transport=transport or ProviderTransport()
        self.live=transport is None or isinstance(self.transport,ProviderTransport)
        self.ledger=cost_ledger
        self.quota_limits=quota_limits
        self.quota_evidence=quota_evidence
        self.pricing_evidence=pricing_evidence
        self.explicit_prices=provider_prices is not None
        self.prices=provider_prices or {'openai':{'input_per_million':2.5,'output_per_million':10},'anthropic':{'input_per_million':2,'output_per_million':10}}
        self.recovery=RecoveryManager.for_worker(self.state_root.parent) if self.live else RecoveryManager(self.state_root/'judge_retries.json')
        self.semaphore=threading.BoundedSemaphore(concurrency)

    @staticmethod
    def _provider_profile(record: dict) -> str:
        # Fingerprints describe backend configuration, including request parameters,
        # not just model weights. Compare each frozen prompt/schema/settings profile.
        return content_hash({key:record[key] for key in
            ('requested_model','request_kind','prompt_hash','schema_hash','settings_hash')})

    def _verify_provider_identity(self,record: dict) -> None:
        model=record['requested_model']
        if record.get('returned_model')!=model:
            raise JudgeError('returned model differs from requested immutable identity',status='model_mismatch')
        version={'returned_model':record['returned_model'],'provider_version':record.get('provider_version')}
        if self.live and model.startswith('gpt-') and not version['provider_version']:
            raise JudgeError('provider fingerprint unavailable',status='model_mismatch')
        registry=self.state_root/'judge_model_registry.json'
        with file_lock(str(registry)+'.lock'):
            saved=read_json(registry) if registry.exists() else {}
            if saved.get('version')=='request_profile_v2':
                profiles=saved['profiles']
            else:
                # Preserve the legacy model-wide registry and reconstruct profile
                # baselines from integrity-verified successes; never reset known
                # profile fingerprints during an implementation repair.
                profiles={}
                for path in self.cache_root.glob('*/*.json'):
                    prior=read_json(path)
                    if prior.get('status')!='success' or prior.get('simulation') is not (not self.live):continue
                    if prior.get('record_hash')!=content_hash({k:v for k,v in prior.items() if k!='record_hash'}):
                        raise JudgeError('cache record integrity corrupt')
                    if prior.get('returned_model')!=prior.get('requested_model'):
                        raise JudgeError('cached returned model mismatch',status='model_mismatch')
                    key=self._provider_profile(prior)
                    identity={'returned_model':prior['returned_model'],'provider_version':prior.get('provider_version')}
                    if key in profiles and profiles[key]!=identity:
                        raise JudgeError('provider version discontinuity within request profile',status='model_mismatch')
                    profiles[key]=identity
                saved={'version':'request_profile_v2','legacy_model_registry':saved,'profiles':profiles}
            key=self._provider_profile(record)
            if key in profiles and profiles[key]!=version:
                raise JudgeError('provider version discontinuity within request profile',status='model_mismatch')
            profiles[key]=version
            atomic_write_json(registry,saved)

    def _reserve_rate(self,provider: str,token_bound: int) -> None:
        if self.quota_limits is None:
            if self.live: raise JudgeError('actual account quota limits unavailable',status='blocked')
            return
        limits=self.quota_limits.get(provider)
        if not isinstance(limits,dict) or any(type(limits.get(k)) is not int or limits[k]<=0 for k in ('requests_per_minute','tokens_per_minute')):
            raise JudgeError('invalid actual account quota limits',status='blocked')
        if self.live and not self.quota_evidence:
            raise JudgeError('current account quota evidence unavailable',status='blocked')
        path=self.state_root/(provider+'_rate_window.json')
        with file_lock(str(path)+'.lock'):
            now=time.time()
            state=read_json(path) if path.exists() else {'reservations':[]}
            events=[row for row in state['reservations'] if row['at']>now-60]
            if len(events)>=limits['requests_per_minute'] or sum(row['tokens'] for row in events)+token_bound>limits['tokens_per_minute']:
                raise JudgeError('account request/token rate allowance unavailable; pause scheduling',status='blocked')
            events.append({'at':now,'tokens':token_bound})
            atomic_write_json(path,{'reservations':events,'quota_limits':limits})

    def conservative_reservation(self,model: str,prompt: str,trace: str,question: str | None=None) -> float:
        provider='openai' if model.startswith('gpt-') else 'anthropic'
        # UTF-8 byte count conservatively bounds byte-level BPE text tokens; include schema/framing.
        input_bound=len((prompt+trace+(question or '')).encode('utf-8'))+4096
        return float(estimate_judge_cost({'input_tokens':input_bound,'output_tokens':64},self.prices[provider]))

    def request(self,kind: str,model: str,prompt: str,trace: str,*,question: str | None=None,settings: dict | None=None,
                schema: dict | None=None,estimated_cost: float | None=None) -> dict:
        if kind not in KINDS: raise ValueError('unknown request kind')
        expected_model='gpt-4o-2024-11-20' if kind in {'evaluation_primary','evaluation_audit'} else 'claude-sonnet-5-5'
        if model!=expected_model: raise JudgeError('request kind/model mismatch',status='model_mismatch')
        if kind in {'rl_pass_b','evaluation_audit'} and question is None:
            raise ValueError('question-aware request requires public question')
        if kind not in {'rl_pass_b','evaluation_audit'} and question is not None:
            raise ValueError('trace-only score must exclude question')
        expected_schema=RL_PASS_B_SCHEMA if kind=='rl_pass_b' else AUDIT_SCHEMA if kind=='evaluation_audit' else SCORE_SCHEMA
        schema=schema or expected_schema
        if schema!=expected_schema: raise ValueError('request schema disagrees with frozen request kind')
        effective=default_settings(model)
        if settings:
            effective.update(settings)
        if effective.get('max_tokens')!=64 or (model.startswith('gpt-') and effective.get('temperature')!=0) or (model.startswith('claude-') and 'temperature' in effective):
            raise ValueError('frozen judge settings violated')
        if model.startswith('claude-') and (effective.get('thinking')!={'type':'between_tools'} or effective.get('output_config',{}).get('effort')!='low'):
            raise ValueError('frozen Sonnet thinking/effort settings violated')
        if not trace.strip(): raise ValueError('missing trace is not an opacity score request')
        if estimated_cost is None and self.live and self.ledger is not None and self.explicit_prices and self.pricing_evidence:
            estimated_cost=self.conservative_reservation(model,prompt,trace,question)
        request={'execution_mode':'production' if self.live else 'simulation','kind':kind,'model':model,'prompt':prompt,'trace':trace,'question':question,'settings':effective,'schema':schema}
        key=content_hash(request)
        path=self.cache_root/kind/(key+'.json')
        provider='openai' if model.startswith('gpt-') else 'anthropic'
        with self.semaphore,file_lock(str(path)+'.lock',timeout=120):
            if path.exists():
                old=read_json(path)
                if old.get('simulation') is not (not self.live): raise JudgeError('cache execution mode mismatch')
                if old.get('record_hash')!=content_hash({k:v for k,v in old.items() if k!='record_hash'}): raise JudgeError('cache record integrity corrupt')
                if old.get('request_hash')!=key: raise JudgeError('cache identity corrupt')
                if old.get('status')=='success':
                    jsonschema.Draft202012Validator(schema).validate(old['parsed'])
                    if _strict_json(old.get('raw_response') or '')!=old['parsed']: raise JudgeError('cache raw/parsed disagreement')
                    if old.get('returned_model')!=model: raise JudgeError('cached returned model mismatch',status='model_mismatch')
                    self._verify_provider_identity(old)
                    return old
                if old.get('status')=='sent' and old.get('raw_response') is not None:
                    # A crash after receiving/persisting the raw response need not purchase another judgment.
                    try:
                        parsed=_strict_json(old['raw_response'])
                        jsonschema.Draft202012Validator(schema).validate(parsed)
                        if old.get('returned_model')!=model: raise JudgeError('incomplete cached response model mismatch',status='model_mismatch')
                        if old.get('finish_reason') in {'length','max_tokens','refusal'}: raise JudgeError('incomplete cached provider response',status='schema_error')
                        self._verify_provider_identity(old)
                        if self.live:
                            if self.ledger is None or not old.get('provider_prices') or not old.get('pricing_evidence'):
                                raise JudgeError('cached response billing reconciliation prerequisites unavailable',status='blocked')
                            actual=estimate_judge_cost(old['usage'],old['provider_prices'][provider])
                            self.ledger.commit(old['attempts'][-1]['attempt_id'],actual,usage=old['usage'])
                            old['cost_usd']=str(actual)
                        old.update(status='success',parsed=parsed,completed_at=timestamp(),elapsed_seconds=old.get('elapsed_seconds',0))
                        old['attempts'][-1]['status']='completed_committed'
                        _persist_record(path,old);notify_artifact(path,old)
                        return old
                    except (ValueError,jsonschema.ValidationError,JudgeError,SpendingBlocked,KeyError,IndexError) as error:
                        old.update(status=getattr(error,'status','schema_error'),error='persisted response reconciliation failed')
                        _persist_record(path,old)
                        return old
            record={'request_hash':key,'request_kind':kind,'provider':provider,'requested_model':model,
                    'prompt_hash':content_hash(prompt),'trace_hash':content_hash(trace),'question_hash':content_hash(question) if question is not None else None,
                    'schema_hash':content_hash(schema),'settings_hash':content_hash(effective),'settings':effective,
                    'status':'sent','started_at':timestamp(),'simulation':not self.live,'attempts':[],
                    'provider_prices':self.prices if self.explicit_prices else None,'pricing_evidence':self.pricing_evidence}
            _persist_record(path,record)
            started=time.monotonic()
            prepared={}
            def prepare_attempt():
                attempt_id=key+'_'+str(len(record['attempts']))+'_'+content_hash(timestamp())[:12]
                if self.live:
                    if not self.explicit_prices or not self.pricing_evidence:
                        raise JudgeError('live provider pricing and current pricing evidence unavailable',status='blocked')
                    if self.ledger is None or estimated_cost is None:
                        raise JudgeError('live judge calls require explicit spend ledger and conservative reservation',status='blocked')
                    try: self.ledger.reserve(attempt_id,estimated_cost,category=kind)
                    except SpendingBlocked as error: raise JudgeError(str(error),status='blocked') from error
                try:
                    self._reserve_rate(provider,len((prompt+trace+(question or '')).encode('utf-8'))+4096+64)
                except JudgeError:
                    if self.live and self.ledger is not None:
                        self.ledger.release(attempt_id,confirmed_unbilled=True)
                    raise
                attempt_record={'attempt_id':attempt_id,'status':'prepared','at':timestamp()}
                record['attempts'].append(attempt_record);_persist_record(path,record)
                prepared.update(attempt_id=attempt_id,attempt_record=attempt_record)
            def attempt():
                attempt_id=prepared['attempt_id'];attempt_record=prepared['attempt_record']
                attempt_record['status']='sent';_persist_record(path,record)
                try:
                    response=self.transport(request)
                except JudgeError as error:
                    attempt_record['status']=error.status;_persist_record(path,record)
                    if error.retryable: raise RetryableFailure(str(error),retry_after=error.retry_after) from error
                    raise
                # Preserve raw provider output before schema parsing, even for ordinary refusals.
                record.update(raw_response=response.get('text'),returned_model=response.get('model'),usage=response.get('usage'),
                              provider_request_id=response.get('request_id'),finish_reason=response.get('finish_reason'),provider_version=response.get('provider_version'))
                _persist_record(path,record)
                if self.live:
                    try:
                        charge=estimate_judge_cost(response.get('usage') or {},self.prices[provider])
                        self.ledger.commit(attempt_id,charge,usage=response['usage'])
                        record['cost_usd']=str(charge)
                    except SpendingBlocked as error: raise JudgeError(str(error),status='blocked') from error
                if response.get('model')!=model:
                    raise JudgeError('returned model differs from requested immutable identity',status='model_mismatch')
                record['provider_version']=response.get('provider_version')
                self._verify_provider_identity(record)
                if response.get('finish_reason') in {'length','max_tokens','refusal'}:
                    raise JudgeError('judge response truncated/refused',status='refusal' if response.get('finish_reason')=='refusal' else 'schema_error')
                try:
                    parsed=_strict_json(response.get('text') or '')
                    jsonschema.Draft202012Validator(schema).validate(parsed)
                except (ValueError,jsonschema.ValidationError,TypeError) as error:
                    raise JudgeError('judge output violates strict response schema',status='schema_error') from error
                attempt_record['status']='completed_committed'
                return parsed
            try:
                parsed=self.recovery.run(key,attempt,before_attempt=prepare_attempt)
                record.update(status='success',parsed=parsed)
            except (JudgeError,RecoveryStopped,RetryableFailure) as error:
                final_status=getattr(error,'status',None)
                if final_status is None and record['attempts']:
                    final_status=record['attempts'][-1].get('status')
                if final_status not in {'timeout','refusal','schema_error','model_mismatch','quota_error','transport_error','blocked'}:
                    final_status='transport_error'
                record.update(status=final_status,error=str(error))
            except Exception as error:
                # Unclassified faults stop this request; don't log a potentially secret-bearing exception body.
                record.update(status='transport_error',error='unclassified '+type(error).__name__)
            record.update(completed_at=timestamp(),elapsed_seconds=time.monotonic()-started)
            _persist_record(path,record)
            if record['status']=='success': notify_artifact(path,record)
            return record
