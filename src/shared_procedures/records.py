"""JSON-compatible boundary validation; schemas don't manufacture scientific evidence."""
from __future__ import annotations
from dataclasses import dataclass,asdict,field
from enum import Enum
from pathlib import Path
from typing import Any,Mapping
import jsonschema
from .identity import content_hash,canonical_json
from .storage import read_json

SCHEMA_ROOT=Path(__file__).resolve().parents[2]/'schemas'
class RecordValidationError(ValueError):
    pass
class OutcomeStatus(str,Enum):
    COMPLETE='complete'
    FAILED='failed'
    INCOMPLETE='incomplete'
    PAUSED='paused'
    INDETERMINATE='indeterminate'
    NOT_ASSESSABLE='not_assessable'

@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    identity: dict
    data: Any
    status: str='complete'
    content_hash: str=field(init=False)
    def __post_init__(self):
        object.__setattr__(self,'content_hash',content_hash(self.data))
        if not self.identity: raise RecordValidationError('artifact identity required')
        canonical_json(asdict(self))
    def to_dict(self) -> dict: return asdict(self)

@dataclass(frozen=True)
class NativeResponse:
    raw_output: str
    reasoning: str
    final_answer: str
    status: str
    parser_version: str
    def to_dict(self) -> dict: return asdict(self)

def validate_record(kind: str,record: Mapping[str,Any],*,schema_root: str | Path=SCHEMA_ROOT) -> dict:
    path=Path(schema_root)/(kind+'.json')
    if not path.is_file(): raise RecordValidationError(f'unknown record kind: {kind}')
    value=dict(record)
    try:
        canonical_json(value)
        schema=read_json(path)
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    except (ValueError,TypeError,jsonschema.ValidationError,jsonschema.SchemaError) as error:
        raise RecordValidationError(f'{kind}: {error.message if hasattr(error,"message") else str(error)}') from error
    if kind=='candidate_prompt' and value['source_content_hash']!=content_hash(value['prompt']):
        raise RecordValidationError('candidate prompt content hash mismatch')
    if kind=='clean_generation':
        if value['raw_hash']!=content_hash(value['raw_generation']):
            raise RecordValidationError('clean raw generation hash mismatch')
        if value.get('clean_trace_hash') is not None and value['clean_trace_hash']!=content_hash({'reasoning':value['reasoning'],'final_answer':value['final_answer']}):
            raise RecordValidationError('clean parsed trace hash mismatch')
    if kind=='judge_record' and value.get('status')=='success':
        request_kind=value['request_kind']
        response_kind='rl_pass_b_response' if request_kind=='rl_pass_b' else 'evaluation_audit_response' if request_kind=='evaluation_audit' else 'score_response'
        validate_record(response_kind,value['parsed'],schema_root=schema_root)
    if kind=='sft_training_record':
        if value['completion']!='<think>'+value['reasoning']+'</think>'+value['final_answer']:
            raise RecordValidationError('SFT completion native boundaries disagree')
        if value['family']!=value['method'] or value['identity'].get('config_hash')!=value['config_hash']:
            raise RecordValidationError('SFT family/configuration identity disagreement')
    return value

def validate_records(kind: str,records: list[dict],*,expected_count: int | None=None,id_field: str='example_id') -> list[dict]:
    if expected_count is not None and len(records)!=expected_count: raise RecordValidationError('incorrect record count')
    checked=[validate_record(kind,row) for row in records]
    ids=[row[id_field] for row in checked]
    if len(set(ids))!=len(ids): raise RecordValidationError('duplicate record identities')
    return checked
