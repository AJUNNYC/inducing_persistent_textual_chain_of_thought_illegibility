#!/usr/bin/env python3
"""Bind passed compatible controls and repaired baselines to certified SFT.

Retain original evidence identities. No model, transform, judge or statistical
control is repeated; incompatible source or scientific settings stop the handoff.
"""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shared_procedures.configuration import configuration_hash, validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, read_json
from shared_procedures.preflight import CheckResult, run_preflights, require_preflights, preflight_requirements
from training.sft.data_handoff import validate_handoff
from training.sft.execution_profile import SFTExecutionProfile
from data_generation_collection.evaluation_data.protocol.analysis_plan.statistical_validation import implementation_hash
from data_generation_collection.evaluation_data.protocol.evaluation_protocol import freeze_evaluation_protocol, validate_protocol


def original_file(commit, name):
    return subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)


def checked_report(path):
    report = read_json(path)
    if report.get('report_hash') != content_hash({k:v for k,v in report.items() if k != 'report_hash'}):
        raise ValueError('Original evidence hash differs: '+str(path))
    return report


def compatible_functions(commit, name, functions):
    old = ast.parse(original_file(commit,name))
    new = ast.parse((ROOT/name).read_text())
    def selected(tree):
        result = {}
        for node in tree.body:
            names = [node.name] if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) else (
                [target.id for target in node.targets if isinstance(target,ast.Name)] if isinstance(node,ast.Assign) else [])
            for name in names:
                if name in functions:
                    result[name] = ast.dump(node,include_attributes=False)
        return result
    if selected(old) != selected(new) or set(selected(new)) != set(functions):
        raise ValueError('Passed control implementation changed: '+name)
    return {function:content_hash(value) for function,value in selected(new).items()}


def identical_files(commit, names):
    result = {}
    for name in names:
        current = (ROOT/name).read_bytes()
        if current != original_file(commit,name):
            raise ValueError('Passed control source changed: '+name)
        result[name] = file_hash(ROOT/name)
    return result


def compatible_sft_numerics(commit, *, loss_implementation='spatial_ce'):
    """Preserve the original masking, CE and reduction beneath new guards.

    Execution profiles can add validated counts or input caching. Their entire
    changed implementation remains bound by the hardware benchmark; this
    narrower source comparison independently protects the scientific objective.
    """
    name = 'src/training/sft/individual_fit/sft_fit.py'
    old = ast.parse(original_file(commit, name))
    new = ast.parse((ROOT/name).read_text())

    flat_expression = ast.parse('token_losses = functional.cross_entropy('
        'shifted_logits.reshape(-1, shifted_logits.shape[-1]), shifted_labels.reshape(-1), '
        'ignore_index=-100, reduction="none").reshape(shifted_labels.shape)').body[0]
    flat_contract = ast.dump(flat_expression, include_attributes=False)

    def contract(tree, *, permit_flat=False):
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        loss = functions['completion_loss']
        selected = {}
        ce_assignments = []
        for node in ast.walk(loss):
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if any(isinstance(target, ast.Tuple) and any(
                        isinstance(item, ast.Name) and item.id == 'shifted_logits' for item in target.elts)
                       for target in node.targets):
                    selected['shift'] = ast.dump(node, include_attributes=False)
                elif 'token_losses' in names:
                    ce_assignments.append(ast.dump(node, include_attributes=False))
                elif 'counts' in names and isinstance(node.value, ast.Call):
                    selected['direct_counts'] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.Return):
                selected['reduction'] = ast.dump(node, include_attributes=False)
        spatial = [value for value in ce_assignments if value != flat_contract]
        if len(spatial) != 1 or (not permit_flat and len(ce_assignments) != 1):
            raise ValueError('Cannot establish the original SFT cross-entropy contract')
        selected['cross_entropy'] = spatial[0]
        if permit_flat and any(value not in (selected['cross_entropy'], flat_contract) for value in ce_assignments):
            raise ValueError('SFT CE dispatch changes the frozen loss objective')
        collate = functions['collate_completion']
        # New guards snapshot the same labels list in a local variable.
        labels_value = ast.parse("list(example['labels'])", mode='eval').body
        class LabelsSnapshot(ast.NodeTransformer):
            def visit_Name(self, node):
                return deepcopy(labels_value) if node.id == 'labels' and isinstance(node.ctx, ast.Load) else node
        for node in ast.walk(collate):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if (isinstance(call.func, ast.Attribute) and call.func.attr == 'append'
                        and isinstance(call.func.value, ast.Subscript)
                        and isinstance(call.func.value.value, ast.Name) and call.func.value.value.id == 'result'
                        and isinstance(call.func.value.slice, ast.Constant)):
                    key = call.func.value.slice.value
                    selected['padding_'+key] = ast.dump(LabelsSnapshot().visit(deepcopy(node)), include_attributes=False)
        required = {'shift', 'cross_entropy', 'direct_counts', 'reduction',
                    'padding_input_ids', 'padding_attention_mask', 'padding_labels'}
        if set(selected) != required:
            raise ValueError('Cannot establish the original SFT mask/loss contract')
        return selected
    if loss_implementation not in ('spatial_ce', 'flat_ce', 'fused_ce'):
        raise ValueError('No accepted scientific loss contract for this execution mode')
    previous, current = contract(old), contract(new)
    if previous != current:
        raise ValueError('Passed SFT mask, cross-entropy or reduction changed')
    bindings = {key: content_hash(value) for key, value in current.items()}
    if loss_implementation != 'spatial_ce':
        wrappers = [node for node in new.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'profile_completion_loss']
        if len(wrappers) != 1:
            raise ValueError('Measured replacement loss dispatch is absent')
        wrapper = wrappers[0]
        fused_expression = ast.parse('token_losses = fused_token_cross_entropy('
            'shifted_logits.reshape(-1, shifted_logits.shape[-1]), shifted_labels.reshape(-1), '
            'ignore_index=-100).reshape(shifted_labels.shape)').body[0]
        fused_contract = ast.dump(fused_expression, include_attributes=False)
        wrapper_contract, ce_dispatch = {}, []
        for node in ast.walk(wrapper):
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if any(isinstance(target, ast.Tuple) and any(
                        isinstance(item, ast.Name) and item.id == 'shifted_logits' for item in target.elts)
                       for target in node.targets):
                    wrapper_contract['shift'] = ast.dump(node, include_attributes=False)
                elif 'counts' in names and isinstance(node.value, ast.Call):
                    wrapper_contract['direct_counts'] = ast.dump(node, include_attributes=False)
                elif 'token_losses' in names:
                    ce_dispatch.append(ast.dump(node, include_attributes=False))
        returns = [node for node in wrapper.body if isinstance(node, ast.Return)]
        if len(returns) != 1:
            raise ValueError('Replacement loss must retain the original per-example reduction')
        wrapper_contract['reduction'] = ast.dump(returns[0], include_attributes=False)
        if (set(wrapper_contract) != {'shift', 'direct_counts', 'reduction'}
                or any(value != previous[key] for key, value in wrapper_contract.items())
                or any(value not in (flat_contract, fused_contract) for value in ce_dispatch)
                or (flat_contract if loss_implementation == 'flat_ce' else fused_contract) not in ce_dispatch):
            raise ValueError('Replacement CE dispatch changed the frozen shift, mask or per-example weighting')
        bindings.update({'replacement_'+key: content_hash(value) for key,value in wrapper_contract.items()})
        bindings[loss_implementation+'_dispatch'] = content_hash(
            flat_contract if loss_implementation == 'flat_ce' else fused_contract)
        if loss_implementation == 'fused_ce':
            bindings['fused_ce_module'] = file_hash(ROOT/'src/training/sft/fused_loss.py')
    return bindings


def accepted_execution_controls(report_path, profile_name, config, certificate, commit, recovery_path=None):
    """Use actual target-condition evidence without expanding its scope."""
    from training.sft.runtime_migration import validate_execution_benchmark
    report_path = Path(report_path)
    report = read_json(report_path)
    plan = report.get('plan')
    if plan is None:
        plan = read_json(report_path.with_name('plan.json'))
    settings = plan.get('profiles', {}).get(profile_name)
    if not isinstance(settings, dict):
        raise ValueError('Requested execution profile is absent from the frozen benchmark plan')
    profile = SFTExecutionProfile.from_dict(settings)
    recovery = read_json(recovery_path) if recovery_path is not None else None
    accepted = validate_execution_benchmark(report, commit, profile, plan=plan,
        profile_name=profile_name, recovery_report=recovery)
    if (accepted.get('accepted') is not True
            or accepted.get('configuration_hash') != configuration_hash(config)
            or accepted.get('handoff_certificate_hash') != certificate['certificate_hash']):
        raise ValueError('Accepted execution evidence names another consumer configuration or certificate')
    condition = accepted.get('condition', {})
    if (condition.get('family') not in ('rot13', 'kazitext')
            or condition.get('model') not in config['models']
            or type(condition.get('level')) is not int or not 1 <= condition['level'] <= 10):
        raise ValueError('Execution evidence lacks one valid frozen SFT condition')
    if accepted.get('recovery_validated') is not True or not accepted.get('original_checkpoint_hash'):
        raise ValueError('Original full-checkpoint paired continuation has not passed')
    configured = config.get('runtime', {}).get('sft_execution_profile')
    if configured is not None and SFTExecutionProfile.from_dict(configured) != profile:
        raise ValueError('Consumer execution profile differs from the measured profile')
    evidence = {'live': True, 'simulation': False, 'control_preflight': True,
                'scientific_results': False, 'condition': condition,
                'scope': 'one_certified_sft_condition', 'accepted_execution': accepted,
                'models_retested': [condition['model']], 'other_models_retested': False,
                'scientific_equivalence_established': False}
    checks = {
        'model_runtime': CheckResult('model_runtime', 'passed',
            'Pinned target-condition runtime measured on the selected hardware; broader model matrix was not repeated.', evidence),
        'loss_mask_gradients': CheckResult('loss_mask_gradients', 'passed',
            'Actual longest native-sequence gradients and paired loss/gradient/update/RNG screen passed for the target condition.', evidence),
        'full_checkpoint_resume': CheckResult('full_checkpoint_resume', 'passed',
            'Original full checkpoint restored and paired whole-update continuation passed without modifying the retained checkpoint.', evidence),
    }
    return accepted, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','certificate','control-config','baseline-root','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--execution-benchmark')
    parser.add_argument('--execution-profile')
    parser.add_argument('--recovery-benchmark')
    args = parser.parse_args(argv)
    if bool(args.execution_benchmark) != bool(args.execution_profile):
        parser.error('--execution-benchmark and --execution-profile must be supplied together')
    if args.recovery_benchmark and not args.execution_benchmark:
        parser.error('--recovery-benchmark requires the execution benchmark')
    config = validate_configuration(read_json(args.config),runtime=True)
    certificate = validate_handoff(read_json(args.certificate),config)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if config['runtime']['implementation_revision'] != commit:
        raise ValueError('Consumer is not the executed frozen source')
    controls = checked_report(ROOT/'docs/evidence/sft_runtime_checks_report.json')
    original_config = read_json(args.control_config)
    if configuration_hash(original_config) != controls['control_configuration_hash']:
        raise ValueError('Original GPU control configuration differs')
    for key in ('models','sft','rl'):
        if config[key] != original_config[key]:
            raise ValueError('Training settings changed after passed controls: '+key)
    gpu_source = controls['source_revision_for_gpu_checks']
    accepted, execution_checks = (accepted_execution_controls(args.execution_benchmark,
        args.execution_profile, config, certificate, commit, args.recovery_benchmark)
        if args.execution_benchmark else (None, {}))
    functions = (['render_sft_completion','tokenize_completion','capture_rng','restore_rng','exposure_order',
                  'SFTRecipe','TARGET_MODULES','LOSS_REDUCTION','MODELS','LEVELS'] if accepted else
                 ['_runtime','render_sft_completion','tokenize_completion','verify_mask_gradients','collate_completion',
                  'completion_loss','capture_rng','restore_rng','train_optimizer_updates','SFTRecipe',
                  'TARGET_MODULES','LOSS_REDUCTION','MODELS','LEVELS'])
    bindings = {'gpu_functions':compatible_functions(gpu_source,
        'src/training/sft/individual_fit/sft_fit.py',
        functions),
        'gpu_control':compatible_functions(gpu_source,'src/shared_procedures/preflight.py',['_run_model_checks']),
        'checkpoint':identical_files(gpu_source,['src/shared_procedures/checkpointing.py',
            'requirements.lock','src/operations/assets.py','src/shared_procedures/identity.py',
            'src/shared_procedures/response_parsing.py','src/shared_procedures/storage.py'])}
    if accepted:
        bindings['scientific_sft_numerics'] = compatible_sft_numerics(gpu_source,
            loss_implementation=accepted['execution_profile'].get('loss_implementation', 'spatial_ce'))
        bindings['accepted_execution'] = accepted
    rl_files = [str(path.relative_to(ROOT)) for path in sorted((ROOT/'src/training/rl_grpo').rglob('*.py'))]
    bindings['grpo'] = identical_files(gpu_source,rl_files)
    judge = checked_report(ROOT/'docs/evidence/judge_profile_guard_live_report.json')
    bindings['judges'] = identical_files(judge['source_revision'],[
        'src/shared_procedures/judge_client.py','src/shared_procedures/cost_accounting.py',
        'src/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/legibility_judging.py',
        'src/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/readable_solution_audit/readable_solution_audit.py',
        'prompts/evaluation_legibility.txt','prompts/evaluation_readable_solution.txt'])
    transform = read_json(ROOT/'docs/evidence/transformation_runtime.json')
    transform_files = [str(path.relative_to(ROOT)) for directory in ('rot13','kazitext')
        for path in sorted((ROOT/'src/data_generation_collection/sft_data'/directory).rglob('*.py'))]
    bindings['transformations'] = identical_files(certificate['producer_commit'],transform_files)
    statistical = checked_report(ROOT/'docs/evidence/statistical_guard_v2_report.json')
    if statistical['implementation_hash'] != implementation_hash():
        raise ValueError('Validated statistical implementation changed')
    baseline_root = Path(args.baseline_root)
    baseline = checked_report(baseline_root/'preflight/baseline_repeatability.json')
    repair = read_json(baseline_root/'repair_receipt.json')
    if (repair.get('receipt_hash') != content_hash({k:v for k,v in repair.items() if k != 'receipt_hash'})
        or repair['report_hash'] != baseline['report_hash'] or repair['new_generations'] != 0
        or repair['preserved_raw_count'] != 7398 or repair['scientific_thresholds_changed'] is not False
        or repair['drive_replication'] != 'VERIFIED' or baseline['ready_for_training'] is not True):
        raise ValueError('Repaired full baseline gate has not passed and been preserved')
    measurement = read_json(baseline_root/'measurement_protocol.json')
    validate_protocol(measurement,allow_draft=True)
    if (repair['protocol_hash'] != measurement['protocol_hash']
        or baseline['evidence']['panel_hash'] != measurement['panel_hash']):
        raise ValueError('Repaired baseline panel/measurement binding differs')
    plan = read_json(ROOT/'docs/evidence/statistical_guard_v2_analysis_plan.json')
    if plan.get('status') != 'frozen' or plan.get('plan_hash') != statistical['analysis_plan_hash']:
        raise ValueError('Analysis plan differs from full-budget validation')
    checks = {}
    for row in controls['checks']:
        if row['name'] in ('model_runtime','loss_mask_gradients','full_checkpoint_resume','durable_storage','mbpp_sandbox','grpo_update'):
            if row['name'] in execution_checks:
                checks[row['name']] = execution_checks[row['name']]
                continue
            copied = deepcopy(row)
            copied['evidence']['compatible_reuse'] = {'original_report_hash':controls['report_hash'],
                'original_control_configuration_hash':controls['control_configuration_hash'],
                'original_source_commit':gpu_source, 'consumer_commit':commit,
                'compatibility_bindings_hash':content_hash(bindings)}
            checks[row['name']] = CheckResult(**copied)
    for row, report_hash in ((judge['canonical_check'],judge['report_hash']),
        (statistical['checks'][0],statistical['report_hash']),(baseline['checks'][0],baseline['report_hash'])):
        copied = deepcopy(row)
        copied['evidence']['compatible_reuse'] = {'original_report_hash':report_hash,
            'consumer_commit':commit,'compatibility_bindings_hash':content_hash(bindings)}
        checks[row['name']] = CheckResult(**copied)
    checks['transformation_runtime'] = CheckResult(**transform)
    checks['clean_data'] = CheckResult('clean_data','passed',
        'Existing independently Drive-audited corpus certified for this consumer; bytes retained.',
        {'live':True,'simulation':False,'record_count':404000,'condition_count':40,
         'certificate_hash':certificate['certificate_hash'],
         'original_drive_readback_sha256':certificate['drive_readback_report_sha256']})
    report = run_preflights(config,checks,output_dir=Path(args.output)/'preflight',scope='training')
    if accepted:
        report['ready_scope'] = accepted['condition']
        report['study_ready'] = False
        report['entire_model_condition_matrix_retested'] = False
        report['report_hash'] = content_hash({key:value for key,value in report.items() if key != 'report_hash'})
        atomic_write_json(Path(args.output)/'preflight'/'report.json', report)
    require_preflights(report,preflight_requirements('training'),expected_configuration_hash=configuration_hash(config))
    protocol = freeze_evaluation_protocol({'items':measurement['items']},measurement['legibility_ids'],
        analysis_plan=plan,preflight_reports=report,configuration_hash=configuration_hash(config),generation_backend='vllm')
    for key in ('generation','prompts','prompt_hashes','parser','scorer','sandbox','judges','opacity','panel_hash','legibility_ids','code_hashes'):
        if protocol['specifications'][key] != measurement['specifications'][key]:
            raise ValueError('Training measurement differs from repaired baseline: '+key)
    atomic_write_json(Path(args.output)/'measurement_protocol.json',protocol)
    receipt = {'status':'READY','source_commit':commit,'configuration_hash':configuration_hash(config),
        'handoff_certificate_hash':certificate['certificate_hash'],'preflight_report_hash':report['report_hash'],
        'baseline_report_hash':baseline['report_hash'],'baseline_protocol_hash':measurement['protocol_hash'],
        'frozen_protocol_hash':protocol['protocol_hash'],'measurement_contract_unchanged':True,
        'source_compatibility_bindings':bindings,'mandatory_gate_count':13,
        'gpu_controls_repeated':False,'statistical_validation_repeated':False,'training_started':False}
    if accepted:
        receipt.update(ready_scope=accepted['condition'], study_ready=False,
            execution_benchmark=accepted, target_sft_controls_repeated=True,
            entire_model_condition_matrix_retested=False)
    receipt['receipt_hash'] = content_hash(receipt)
    atomic_write_json(Path(args.output)/'readiness_receipt.json',receipt)
    print(__import__('json').dumps({k:v for k,v in receipt.items() if k != 'source_compatibility_bindings'}),flush=True)


if __name__ == '__main__':
    main()
