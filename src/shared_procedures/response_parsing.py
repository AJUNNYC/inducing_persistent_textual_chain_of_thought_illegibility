"""Frozen mechanical parsers. Raw strings are never repaired or inferred."""
from __future__ import annotations
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Any

PARSER_VERSION = 'native_think_strict_v1'
NUMERIC_VERSION = 'rational_decimal_v1'

def parse_native_output(text: str) -> dict[str, Any]:
    result = {'reasoning': '', 'final_answer': '', 'status': 'malformed', 'reason': None, 'parser_version': PARSER_VERSION}
    if not isinstance(text, str):
        raise TypeError('native output must be text')
    if text.count('<think>') != 1 or text.count('</think>') != 1:
        result['reason'] = 'missing_or_ambiguous_native_boundaries'
        return result
    opening = text.index('<think>')
    closing = text.index('</think>')
    if text[:opening].strip() or closing < opening:
        result['reason'] = 'invalid_native_order'
        return result
    result.update(reasoning=text[opening+7:closing], final_answer=text[closing+8:])
    if not result['reasoning'].strip() or not result['final_answer'].strip():
        result['reason'] = 'empty_reasoning_or_final'
    else:
        result.update(status='valid', reason=None)
    return result

_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

def normalize_numeric_answer(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    value = text.strip().replace('\u2212', '-')
    if len(value)>256:
        return None
    for exponent in re.findall(r'[eE]([+-]?\d+)',value):
        if len(exponent.lstrip('+-'))>4 or abs(int(exponent))>1000:
            return None
    if value.endswith(('.', '!', '?', ';')):
        value = value[:-1].rstrip()
    if ',' in value:
        if not re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', value):
            return None
        value = value.replace(',', '')
    if re.fullmatch(_NUMBER, value):
        try:
            number = Fraction(Decimal(value))
        except (InvalidOperation, ValueError, OverflowError):
            return None
    elif re.fullmatch(rf'({_NUMBER})\s*/\s*({_NUMBER})', value):
        numerator, denominator = value.split('/')
        try:
            number = Fraction(Decimal(numerator.strip())) / Fraction(Decimal(denominator.strip()))
        except (InvalidOperation, ZeroDivisionError, ValueError, OverflowError):
            return None
    else:
        return None
    return str(number.numerator) if number.denominator == 1 else f'{number.numerator}/{number.denominator}'

def parse_rl_output(text: str, *, truncated: bool = False) -> dict[str, Any]:
    result = {'reasoning': '', 'final_answer': '', 'numeric_answer': None, 'status': 'invalid', 'reason': None,
              'parser_version': 'rl_final_box_v1'}
    result['truncated'] = bool(truncated)
    if text.count('Final answer:') != 1:
        result['reason'] = 'missing_or_ambiguous_final_marker'
        return result
    match = re.search(r'(?m)^Final answer:\s*(?:\\)?boxed\{([^{}]+)\}\s*[.!]?\s*\Z', text)
    if match is None:
        result['reason'] = 'invalid_final_line'
        return result
    reasoning = text[:match.start()]
    # Native delimiters are framing, not reasoning; strip only exact boundaries.
    if reasoning.startswith('<think>'):
        reasoning = reasoning[7:]
    if reasoning.rstrip().endswith('</think>'):
        reasoning = reasoning.rstrip()[:-8]
    numeric = normalize_numeric_answer(match.group(1))
    result.update(reasoning=reasoning, final_answer=text[match.start():], numeric_answer=numeric)
    if not reasoning.strip() or numeric is None:
        result['reason'] = 'empty_reasoning_or_invalid_number'
    else:
        result.update(status='valid', reason=None)
    return result

def parse_rl_answer(text: str) -> str | None:
    return parse_rl_output(text)['numeric_answer']

def extract_choice_label(text: str, *, labels: str = 'ABCD') -> str | None:
    match = re.fullmatch(r'\s*(?:Answer:\s*)?\(?(['+re.escape(labels)+r'])\)?[.]?\s*', text)
    return match.group(1) if match else None

def extract_final_code(text: str) -> str | None:
    if text.strip().startswith('```'):
        match = re.fullmatch(r'\s*```(?:python)?\r?\n(.*?)\r?\n```\s*', text, flags=re.DOTALL)
        return match.group(1) if match and '```' not in match.group(1) else None
    return text if text.strip() and '```' not in text else None
