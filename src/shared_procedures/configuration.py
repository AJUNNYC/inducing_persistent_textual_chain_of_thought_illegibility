"""Scientific configuration validation and separate live readiness checks."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import re
import os
import shlex
from typing import Any
import yaml
from .identity import content_hash
from .storage import atomic_write_json

class ConfigurationError(ValueError):
    pass

MODEL_IDS = {'7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', '14b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'}
KAZITEXT_ALPHAS = ['0.200','0.301','0.454','0.684','1.031','1.553','2.339','3.524','5.310','8.000']
PROJECTION_TARGETS = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
SFT_TRAINING_BUDGETS = {'legacy_two_epoch_v1': 2, 'one_epoch_pilot_v1': 1,
                        'efficient_one_epoch_v1': 1}
SFT_BUDGET_OVERRIDES = {'efficient_one_epoch_v1': {
    'sft.quantization': 'bf16', 'sft.load_in_4bit': False, 'sft.double_quantization': False,
    'sft.optimizer': 'adamw', 'sft.gradient_checkpointing': False}}
FIXED = {
    'seed': 42,
    'generation.temperature': 0.6, 'generation.top_p': 0.95, 'generation.max_new_tokens': 3072,
    'generation.system_prompt': None,
    'data.sft_train_count': 10000, 'data.sft_validation_count': 100, 'data.rl_count': 800,
    'datasets.lmsys.id': 'lmsys/lmsys-chat-1m',
    'datasets.gsm8k.id': 'openai/gsm8k',
    'datasets.mmlu.id': 'edinburgh-dawg/mmlu-redux-2.0',
    'datasets.arc.id': 'allenai/ai2_arc',
    'datasets.arc.configuration': 'ARC-Easy',
    'datasets.mbpp.id': 'google-research-datasets/mbpp',
    'datasets.mbpp.configuration': 'sanitized',
    'data.prompt_max_tokens': 200, 'data.minimum_eligible_words': 3,
    'interventions.rot13.fractions': ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],
    'interventions.kazitext.alphas': KAZITEXT_ALPHAS,
    'interventions.kazitext.revision': '435f68748fdded6d79446b8ab6bdf76fbde88818',
    'interventions.kazitext.profile': 'profiles/dev/en_clean_second_learners.json',
    'interventions.kazitext.profile_sha256': '8c723dd5925fc872b4ee8a5191f585dee95d1146e849fc80f5e84f3cd0f62d73',
    'interventions.kazitext.lang': 'en', 'interventions.kazitext.alpha_std': 0,
    'interventions.kazitext.alpha_uniformity_prob': 0, 'interventions.kazitext.beta': 0,
    'interventions.kazitext.no_word_order': True, 'interventions.kazitext.no_error_sentence_boost': 0,
    'interventions.kazitext.spelling_detailed_ratio': 0.3, 'interventions.kazitext.strip_all_diacritics': False,
    'interventions.kazitext.other_error_aspects_enabled': True,
    'interventions.kazitext.udpipe_model': 'english-ewt-ud-2.5-191206.udpipe',
    'sft.quantization': 'nf4', 'sft.load_in_4bit': True, 'sft.double_quantization': True,
    'sft.compute_dtype': 'bfloat16', 'sft.lora_rank': 8, 'sft.lora_alpha': 16, 'sft.lora_dropout': 0.05,
    'sft.target_modules': PROJECTION_TARGETS, 'sft.learning_rate': 0.0002,
    'sft.optimizer': 'paged_adamw_32bit', 'sft.scheduler': 'cosine', 'sft.warmup_ratio': 0.03,
    'sft.epochs': 2, 'sft.effective_batch_size': 16, 'sft.microbatch_size': 1,
    'sft.gradient_accumulation_steps': 16, 'sft.max_seq_length': 4096,
    'sft.weight_decay': 0, 'sft.max_grad_norm': 0.3, 'sft.gradient_checkpointing': True,
    'sft.loss_reduction': 'mean_supervised_tokens_per_example_then_mean_examples',
    'sft.training_template_version': 'native_training_template_v1',
    'sft.early_stopping': False, 'sft.checkpoint_interval': 100, 'sft.loss_mask': 'completion_only',
    'rl.steps': 100, 'rl.questions_per_step': 8, 'rl.completions_per_question': 4,
    'rl.temperature': 1.0, 'rl.top_p': 1.0, 'rl.top_k': 0, 'rl.repetition_penalty': 1.0,
    'rl.max_new_tokens': 2048, 'rl.lora_rank': 32, 'rl.lora_alpha': 32,
    'rl.learning_rate': 0.000005, 'rl.beta': 0.01, 'rl.opacity_weight': 0.5,
    'rl.checkpoint_interval': 10, 'rl.loss_type': 'dr_grpo', 'rl.scale_rewards': 'group',
    'rl.num_iterations': 1, 'rl.importance_sampling_level': 'token', 'rl.epsilon': 0.2,
    'rl.epsilon_high': 0.2, 'rl.reference_policy': 'frozen_original',
    'rl.mask_truncated_completions': False, 'rl.use_vllm': False, 'rl.reshuffle': False,
    'rl.entropy_bonus': 0, 'evaluation.capability_counts': {'gsm8k':300,'mmlu':399,'arc':331,'mbpp':203},
    'evaluation.legibility_per_domain': 50, 'evaluation.seed': 42, 'evaluation.repeatability_seeds': [43,44],
    'evaluation.mbpp.cpu_count': 1, 'evaluation.mbpp.memory_bytes': 1073741824,
    'evaluation.mbpp.cpu_seconds': 5, 'evaluation.mbpp.wall_seconds': 10,
    'judges.primary_model': 'gpt-4o-2024-11-20', 'judges.secondary_model': 'claude-sonnet-5-5',
    'judges.reward_model': 'claude-sonnet-5-5', 'judges.max_tokens': 64,
    'judges.gpt_temperature': 0, 'judges.sonnet_thinking_type': 'between_tools',
    'judges.sonnet_effort': 'low', 'judges.max_concurrency': 8,
    'recovery.max_operation_retries': 3, 'recovery.max_worker_retries': 100,
    'analysis.bootstrap_draws': 10000, 'analysis.seed': 42, 'analysis.total_windows': 216,
    'analysis.coverage': 0.95,
    'criteria.minimum_window_length': 3, 'criteria.per_level_retention': 0.8,
    'criteria.joint_retention': 0.6, 'criteria.qualified_opacity': 0.5,
    'criteria.opacity_increase': 0.2, 'criteria.primary_score_threshold': 6,
    'criteria.transfer_nonmath_domains': 2,
}

def _get(config: dict, key: str) -> Any:
    current: Any = config
    for segment in key.split('.'):
        if not isinstance(current, dict) or segment not in current:
            raise ConfigurationError(f'missing scientific field: {key}')
        current = current[segment]
    return current

def validate_configuration(config: dict, *, runtime: bool = False) -> dict:
    if not isinstance(config, dict):
        raise ConfigurationError('configuration must be a mapping')
    for key, expected in FIXED.items():
        if key == 'sft.epochs':
            expected = sft_budget_epochs(config)
        elif key in SFT_BUDGET_OVERRIDES.get(sft_training_budget(config), {}):
            expected = SFT_BUDGET_OVERRIDES[sft_training_budget(config)][key]
        actual = _get(config, key)
        if type(expected) is int and type(actual) is not int:
            raise ConfigurationError(f'{key} must be integer')
        if isinstance(expected, bool) and type(actual) is not bool:
            raise ConfigurationError(f'{key} must be boolean')
        if actual != expected:
            raise ConfigurationError(f'frozen setting violated: {key}')
    sft_execution_selection(config)
    if set(config.get('models', {})) != set(MODEL_IDS):
        raise ConfigurationError('exactly 7b and 14b construction models required')
    for size, name in MODEL_IDS.items():
        if config['models'][size].get('id') != name:
            raise ConfigurationError('construction model substitution forbidden')
    _reject_secrets(config)
    if runtime:
        missing = runtime_requirements(config)
        if missing:
            raise ConfigurationError('unresolved runtime prerequisites: '+', '.join(missing))
    return deepcopy(config)


def sft_training_budget(config: dict) -> str:
    """Historical snapshots retain their original two-epoch training identity."""
    budget = config.get('sft', {}).get('training_budget', 'legacy_two_epoch_v1')
    if not isinstance(budget, str) or budget not in SFT_TRAINING_BUDGETS:
        raise ConfigurationError('unknown sft.training_budget')
    return budget


def sft_budget_epochs(config: dict) -> int:
    return SFT_TRAINING_BUDGETS[sft_training_budget(config)]


def sft_declared_recipe_changes(config: dict) -> dict:
    """Only these named, validated settings differ from the historical recipe."""
    values = {'epochs': sft_budget_epochs(config)}
    for key in SFT_BUDGET_OVERRIDES.get(sft_training_budget(config), {}):
        values[key.split('.', 1)[1]] = _get(config, key)
    return values


def sft_execution_selection(config: dict) -> list[dict]:
    """Validate the declared execution scope without reducing the certified corpus."""
    selection = config.get('sft', {}).get('selection')
    if 'selection' not in config.get('sft', {}):
        models, families, levels = ['7b', '14b'], ['rot13', 'kazitext'], list(range(1, 11))
    else:
        if not isinstance(selection, dict) or set(selection) != {'models', 'families', 'levels'}:
            raise ConfigurationError('sft.selection requires models, families and levels')
        models, families, levels = (selection[name] for name in ('models', 'families', 'levels'))
        for name, values, allowed in (
                ('models', models, ('7b', '14b')),
                ('families', families, ('rot13', 'kazitext')),
                ('levels', levels, tuple(range(1, 11)))):
            if (not isinstance(values, list) or not values
                    or any(type(value) is not (int if name == 'levels' else str) for value in values)
                    or len(set(values)) != len(values) or any(value not in allowed for value in values)):
                raise ConfigurationError('invalid sft.selection.' + name)
    return [{'family': family, 'model_id': model, 'level': level}
            for family in ('rot13', 'kazitext') if family in families
            for model in ('7b', '14b') if model in models
            for level in range(1, 11) if level in levels]

def _reject_secrets(value: Any, prefix: str = '') -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).lower()
            if name in {'api_key','access_token','refresh_token','password','client_secret','authorization'} and item:
                raise ConfigurationError(f'embedded credential forbidden: {prefix+str(key)}')
            _reject_secrets(item, prefix+str(key)+'.')
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item,prefix)

def runtime_requirements(config: dict) -> list[str]:
    missing = []
    for size in MODEL_IDS:
        for name in ('revision','tokenizer_revision'):
            value = config['models'][size].get(name)
            if not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{40}',value):
                missing.append(f'models.{size}.{name}')
    for name, source in config.get('datasets', {}).items():
        revision = source.get('revision')
        if not isinstance(revision,str) or not re.fullmatch(r'[0-9a-f]{40}',revision):
            missing.append(f'datasets.{name}.revision')
    kazi = config['interventions']['kazitext']
    if not re.fullmatch(r'[0-9a-f]{64}', kazi.get('udpipe_sha256') or ''):
        missing.append('interventions.kazitext.udpipe_sha256')
    provenance=kazi.get('aspell_dictionary_provenance')
    if not isinstance(provenance,dict) or not isinstance(provenance.get('settings'),dict) or not provenance.get('settings') or not isinstance(provenance.get('files'),list) or not provenance.get('files'):
        missing.append('interventions.kazitext.aspell_dictionary_provenance')
    else:
        names=set()
        for item in provenance['files']:
            if not isinstance(item,dict) or set(item)!={'name','sha256'} or not isinstance(item.get('name'),str) or not item['name'] or '/' in item['name'] or item['name'] in names or not re.fullmatch(r'[0-9a-f]{64}',item.get('sha256') or ''):
                missing.append('interventions.kazitext.aspell_dictionary_provenance')
                break
            names.add(item['name'])
    if not re.fullmatch(r'[0-9a-f]{64}',config.get('runtime',{}).get('dependency_lock_hash') or ''):
        missing.append('runtime.dependency_lock_hash')
    if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})',config.get('runtime',{}).get('implementation_revision') or ''):
        missing.append('runtime.implementation_revision')
    return missing

def load_configuration(path: str | Path, *, require_runtime: bool = False) -> dict:
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader,node,deep=False):
        result = {}
        for key_node,value_node in node.value:
            key = loader.construct_object(key_node,deep=deep)
            if key in result:
                raise ConfigurationError(f'duplicate configuration key: {key}')
            result[key] = loader.construct_object(value_node,deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    config_path=Path(path)
    load_private_environment(config_path.parent/'.env')
    value = yaml.load(config_path.read_text(encoding='utf-8'),Loader=UniqueLoader)
    value=validate_configuration(value,runtime=require_runtime)
    for environment,key in {'DATA_ROOT':'data_root','ARTIFACT_ROOT':'artifact_root','MODEL_CACHE_ROOT':'cache_root'}.items():
        if os.getenv(environment):
            value.setdefault('runtime',{})[key]=os.environ[environment]
    return value

# Explicit operational exclusions: updates cannot change model/data/decoder/loss/reward/measurement identity.
OPERATIONAL_IDENTITY_EXCLUSIONS = (
    'spending.authorized','spending.authorized_usd','spending.authorization_evidence',
    'spending.provider_prices','spending.pricing_evidence','spending.quota_limits','spending.quota_evidence',
    'runtime.data_root','runtime.artifact_root','runtime.cache_root','runtime.credential_env',
    'runtime.durability_evidence','runtime.preflight_evidence_path',
    'judges.quota_limits','judges.quota_evidence',
    'operations.udpipe_path','operations.aspell_dictionary_files',
)

def scientific_configuration(config: dict) -> dict:
    result=validate_configuration(config)
    for path in OPERATIONAL_IDENTITY_EXCLUSIONS:
        parts=path.split('.')
        current=result
        for part in parts[:-1]:
            if not isinstance(current,dict): break
            current=current.get(part,{})
        if isinstance(current,dict):current.pop(parts[-1],None)
    return result

def configuration_hash(config: dict) -> str:
    return content_hash(scientific_configuration(config))

def snapshot_hash(config: dict) -> str:
    return content_hash(validate_configuration(config))

def freeze_configuration(config: dict, path: str | Path) -> dict:
    validated = validate_configuration(config,runtime=True)
    record = {'configuration':scientific_configuration(validated),'configuration_hash':configuration_hash(validated),'status':'frozen'}
    target=Path(path)
    from .storage import read_json
    if target.exists() and read_json(target) != record:
        raise ConfigurationError('cannot overwrite a different frozen scientific configuration')
    atomic_write_json(target,record)
    audit_hash=snapshot_hash(validated)
    audit_path=target.with_name(target.stem+'_snapshot_'+audit_hash+target.suffix)
    audit={'configuration':validated,'configuration_hash':record['configuration_hash'],'snapshot_hash':audit_hash,'status':'resolved_snapshot'}
    if audit_path.exists() and read_json(audit_path)!=audit:raise ConfigurationError('resolved snapshot integrity failure')
    atomic_write_json(audit_path,audit)
    return {**record,'snapshot_hash':audit_hash,'snapshot_path':str(audit_path)}

def sft_matrix(config: dict) -> list[dict]:
    validate_configuration(config)
    return [{'family':row['family'],'model':row['model_id'],'level':f"level_{row['level']:02d}"}
            for row in sft_execution_selection(config)]


def load_private_environment(path: str | Path) -> None:
    """Read optional private .env without shell expansion or embedding values in artifacts.

    Existing injected variables take precedence. Values never appear in logs/config snapshots.
    """
    source=Path(path)
    if not source.exists(): return
    for index,line in enumerate(source.read_text(encoding='utf-8').splitlines(),1):
        line=line.strip()
        if not line or line.startswith('#'): continue
        if line.startswith('export '): line=line[7:]
        if '=' not in line: raise ConfigurationError(f'invalid .env assignment on line {index}')
        key,value=line.split('=',1)
        key=key.strip()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',key): raise ConfigurationError(f'invalid .env key on line {index}')
        lexer=shlex.shlex(value,posix=True);lexer.whitespace_split=True;lexer.commenters='#'
        pieces=list(lexer)
        if len(pieces)>1: raise ConfigurationError(f'quote whitespace in .env value on line {index}')
        os.environ.setdefault(key,pieces[0] if pieces else '')
