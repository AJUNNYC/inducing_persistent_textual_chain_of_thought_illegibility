#!/usr/bin/env python3
"""Measure model-scoped native baseline controls on replacement hardware.

This is fresh eager/gradient-checkpointed SFT qualification, not an optimization
trial or an authorization to migrate an existing optimizer lineage. Existing
data, judge, transformation, baseline and statistical evidence remain reusable.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shared_procedures.configuration import configuration_hash, validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.preflight import _run_model_checks
from shared_procedures.storage import atomic_write_json, read_json
from training.sft.execution_profile import SFTExecutionProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', choices=('7b', '14b'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = validate_configuration(read_json(args.config), runtime=True)
    if config['sft'].get('training_budget') == 'efficient_one_epoch_v1':
        raise ValueError('Published execution is prepared with reconcile_sft_launch.py --published-method and initialized by run_certified_sft.py')
    checks = _run_model_checks(config, include_training=True, model_ids=(args.model,))
    import torch
    records = {name: check.to_dict() for name, check in checks.items()}
    for row in records.values():
        row['detail'] = 'Selected-model native baseline control completed.'
    sources = [*sorted((ROOT/'src/training/sft').rglob('*.py')),
               ROOT/'src/shared_procedures/preflight.py',
               ROOT/'src/shared_procedures/checkpointing.py', ROOT/'requirements.lock',
               Path(__file__)]
    report = {'schema_version': 'sft_baseline_runtime_controls_v1',
              'status': 'PASSED' if all(x['status']=='passed' for x in records.values()) else 'FAILED',
              'source_code_commit': config['runtime']['implementation_revision'],
              'consumer_configuration_hash': configuration_hash(config),
              'control_configuration_hash': configuration_hash(config),
              'model_ids': [args.model], 'models': {args.model:config['models'][args.model]},
              'execution_profile': SFTExecutionProfile().to_dict(),
              'dependency_hash': file_hash(ROOT/'requirements.lock'),
              'source_file_hashes': {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
              'hardware': {'gpu': torch.cuda.get_device_name(0),
                           'compute_capability': list(torch.cuda.get_device_capability(0)),
                           'vram_bytes': torch.cuda.get_device_properties(0).total_memory},
              'checks': records, 'training_started': False,
              'optimization_activated': False, 'parent_checkpoint_modified': False}
    report['report_hash'] = content_hash(report)
    atomic_write_json(args.output, report)
    print(__import__('json').dumps({'status': report['status'], 'model_ids': report['model_ids'],
                                 'report_hash': report['report_hash']}), flush=True)
    return 0 if report['status']=='PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
