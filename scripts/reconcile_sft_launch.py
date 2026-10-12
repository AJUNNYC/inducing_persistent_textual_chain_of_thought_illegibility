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


def spatial_native_forward_contract():
    """Bind the original spatial native forward and explicit backward source."""
    path = ROOT/'src/training/sft/spatial_loss.py'
    tree = ast.parse(path.read_text())
    forwards = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == 'forward']
    if len(forwards) != 1:
        raise ValueError('Spatial loss lacks one explicit native forward')
    wanted = ast.parse('''
log_probabilities = torch.log_softmax(logits, dim=1, dtype=logits.dtype)
losses = functional.nll_loss(log_probabilities, targets, ignore_index=ignore_index, reduction="none")
return losses
''').body
    selected = [node for node in forwards[0].body if isinstance(node, ast.Return)
                or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id in ('log_probabilities', 'losses') for target in node.targets)]
    expected = [ast.dump(node, include_attributes=False) for node in wanted]
    current = [ast.dump(node, include_attributes=False) for node in selected]
    if current != expected:
        raise ValueError('Spatial loss changed the original FP32 spatial log-softmax/NLL forward')
    derivative = ast.parse('derivative = target_gradient - libdevice.exp(log_probability) * summed_gradient').body[0]
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'derivative' for target in node.targets)]
    if (len(matches) != 1 or ast.dump(matches[0], include_attributes=False)
            != ast.dump(derivative, include_attributes=False)):
        raise ValueError('Spatial loss changed the source-bound backward expression')
    return {'spatial_loss_module': file_hash(path),
            'native_spatial_forward': content_hash(current),
            'native_spatial_backward_expression': content_hash(ast.dump(derivative, include_attributes=False))}


def scheduled_spatial_forward_contract():
    """Bind the explicit serial-class recurrence and original short-time route.

    This source contract protects the declared FP32 accumulation order. Actual
    CUDA equivalence and incremental speed still require the paired benchmark.
    """
    path = ROOT/'src/training/sft/scheduled_spatial_loss.py'
    tree = ast.parse(path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    loader = functions.get('_load_scheduled_kernels')
    if loader is None:
        raise ValueError('Scheduled spatial loss lacks its explicit kernel loader')
    kernels = {node.name: node for node in loader.body if isinstance(node, ast.FunctionDef)}
    if set(kernels) != {'forward', 'backward'}:
        raise ValueError('Scheduled spatial loss lacks its two explicit kernels')
    forward = ast.parse('''
batch = tl.program_id(0).to(tl.int64)
times = tl.program_id(1) * block_size + tl.arange(0, block_size)
present = times < n_times
row_offset = batch * n_columns * n_times + times
token_offset = batch * n_times + times
maximum = tl.full((block_size,), -3.4028234663852886e38, tl.float32)
for class_index in range(0, n_columns):
    values = tl.load(logits + row_offset + class_index.to(tl.int64) * n_times, present, other=-3.4028234663852886e38)
    maximum = tl.where(maximum < values, values, maximum)
normalization = tl.full((block_size,), 0.0, tl.float32)
for class_index in range(0, n_columns):
    values = tl.load(logits + row_offset + class_index.to(tl.int64) * n_times, present, other=-3.4028234663852886e38)
    normalization = normalization + libdevice.exp(values - maximum)
log_sum = libdevice.log(normalization)
target = tl.load(targets + token_offset, present, other=ignore_index)
target_value = tl.load(logits + row_offset + target * n_times, present & (target != ignore_index), other=0.0)
loss = tl.where(target == ignore_index, 0.0, -(target_value - maximum - log_sum))
tl.store(losses + token_offset, loss, present)
tl.store(maxima + token_offset, maximum, present)
tl.store(log_sums + token_offset, log_sum, present)
''').body
    backward = ast.parse('''
offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
present = offsets < n_elements
times = offsets % n_times
class_index = (offsets // n_times) % n_columns
batch = offsets // (n_times * n_columns)
token_offset = batch * n_times + times
values = tl.load(logits + offsets, present, other=0.0)
maximum = tl.load(maxima + token_offset, present, other=0.0)
log_sum = tl.load(log_sums + token_offset, present, other=0.0)
log_probability = values - maximum - log_sum
target = tl.load(targets + token_offset, present, other=ignore_index)
row_scale = tl.load(row_scales + token_offset, present, other=0.0)
summed_gradient = tl.where(target == ignore_index, 0.0, -row_scale)
target_gradient = tl.where(class_index == target, summed_gradient, 0.0)
derivative = target_gradient - libdevice.exp(log_probability) * summed_gradient
tl.store(output + offsets, derivative, present)
''').body
    dumps = lambda nodes: [ast.dump(node, include_attributes=False) for node in nodes]
    if dumps(kernels['forward'].body) != dumps(forward):
        raise ValueError('Scheduled spatial loss changed the FP32 serial-class forward recurrence')
    if dumps(kernels['backward'].body) != dumps(backward):
        raise ValueError('Scheduled spatial loss changed the source-bound reconstructed backward')
    wanted_route = ast.parse('''
if type(n_times) is not int or n_times < 1:
    raise ValueError("Spatial CE time must be a positive integer")
return "original_spatial" if n_times <= ORIGINAL_FORWARD_MAX_TIME else "serial_token_blocks"
''').body
    route = functions.get('forward_route')
    route_body = [] if route is None else [node for node in route.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str))]
    constants = {target.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant) for target in node.targets if isinstance(target, ast.Name)}
    if (dumps(route_body) != dumps(wanted_route) or constants.get('ORIGINAL_FORWARD_MAX_TIME') != 64
            or constants.get('TOKEN_BLOCK_SIZE') != 64):
        raise ValueError('Scheduled spatial loss changed its declared shape route')
    autograd = functions.get('_autograd_function')
    classes = [] if autograd is None else [node for node in autograd.body if isinstance(node, ast.ClassDef)]
    methods = [] if len(classes) != 1 else [node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == 'forward']
    route_assignment = ast.parse('ctx.route = forward_route(logits.shape[2])').body[0]
    native_test = ast.parse('ctx.route == "original_spatial"', mode='eval').body
    branches = [] if len(methods) != 1 else [node for node in methods[0].body if isinstance(node, ast.If)]
    if (len(methods) != 1 or not methods[0].body
            or ast.dump(methods[0].body[0], include_attributes=False) != ast.dump(route_assignment, include_attributes=False)
            or len(branches) != 1 or ast.dump(branches[0].test, include_attributes=False)
                != ast.dump(native_test, include_attributes=False) or branches[0].orelse):
        raise ValueError('Scheduled spatial loss changed its explicit native-forward route')
    native = [node for node in branches[0].body if isinstance(node, ast.Return)
        or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id in ('log_probabilities', 'losses') for target in node.targets)]
    wanted_native = ast.parse('''
log_probabilities = torch.log_softmax(logits, dim=1, dtype=logits.dtype)
losses = functional.nll_loss(log_probabilities, targets, ignore_index=ignore_index, reduction="none")
return losses
''').body
    if dumps(native) != dumps(wanted_native):
        raise ValueError('Scheduled spatial loss changed the original short-time native spatial forward')
    bindings = spatial_native_forward_contract()
    bindings.update({'scheduled_spatial_loss_module': file_hash(path),
        'scheduled_spatial_class_serial_forward': content_hash(dumps(forward)),
        'scheduled_spatial_reconstructed_backward': content_hash(dumps(backward)),
        'scheduled_spatial_shape_route': content_hash(dumps(wanted_route))})
    return bindings


def causal_gradient_layout_contract():
    """Protect accepted FP32 forward, causal targets and complete gradient."""
    from training.sft.runtime_migration import ACCEPTED_SCHEDULED_HELPER_SHA256
    bindings = scheduled_spatial_forward_contract()
    if bindings['scheduled_spatial_loss_module'] != ACCEPTED_SCHEDULED_HELPER_SHA256:
        raise ValueError('Causal gradient layout changed the previously accepted scheduled helper')
    path = ROOT/'src/training/sft/causal_gradient_loss.py'
    tree = ast.parse(path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    loader = functions.get('_load_gradient_kernels')
    kernels = {} if loader is None else {node.name: node for node in loader.body if isinstance(node, ast.FunctionDef)}
    autograd = functions.get('_autograd_function')
    classes = [] if autograd is None else [node for node in autograd.body if isinstance(node, ast.ClassDef)]
    methods = {} if len(classes) != 1 else {node.name: node for node in classes[0].body if isinstance(node, ast.FunctionDef)}
    if set(kernels) != {'backward', 'zero_last_row'} or set(methods) != {'forward', 'backward'}:
        raise ValueError('Causal gradient layout lacks explicit forward, backward and zero-tail kernels')
    expected = {
        'forward_delegation': (methods['forward'], '''
ctx.full_shape = tuple(logits.shape)
shifted_logits = logits[:, :-1, :].float()
return _scheduled_autograd_function().forward(ctx, shifted_logits.transpose(1, 2), targets, ignore_index)
'''),
        'tiled_backward': (kernels['backward'], '''
batch = tl.program_id(0).to(tl.int64)
classes = (tl.program_id(1) * class_tile + tl.arange(0, class_tile))[:, None]
times = (tl.program_id(2) * time_tile + tl.arange(0, time_tile))[None, :]
present = (classes < n_columns) & (times < n_times)
source_offset = (batch * n_columns + classes) * n_times + times
token_offset = batch * n_times + times
log_probability = tl.load(values + source_offset, present, other=0.0)
if reconstruct:
    maximum = tl.load(maxima + token_offset, times < n_times, other=0.0)
    log_sum = tl.load(log_sums + token_offset, times < n_times, other=0.0)
    log_probability = log_probability - maximum - log_sum
target = tl.load(targets + token_offset, times < n_times, other=ignore_index)
row_scale = tl.load(row_scales + token_offset, times < n_times, other=0.0)
summed_gradient = tl.where(target == ignore_index, 0.0, -row_scale)
target_gradient = tl.where(classes == target, summed_gradient, 0.0)
derivative = target_gradient - libdevice.exp(log_probability) * summed_gradient
destination_offset = (batch * full_times + times) * n_columns + classes
tl.store(output + destination_offset, derivative, present)
'''),
        'zero_tail': (kernels['zero_last_row'], '''
batch = tl.program_id(0).to(tl.int64)
classes = tl.program_id(1) * block_size + tl.arange(0, block_size)
offsets = (batch * full_times + full_times - 1) * n_columns + classes
tl.store(output + offsets, 0.0, classes < n_columns)
'''),
        'full_gradient_dispatch': (methods['backward'], '''
saved = ctx.saved_tensors
values, targets = saved[:2]
row_scales = row_scales.contiguous()
output = torch.empty(ctx.full_shape, dtype=torch.float32, device=values.device)
reconstruct = ctx.route == "serial_token_blocks"
maxima, log_sums = saved[2:] if reconstruct else (row_scales, row_scales)
kernel, zero_last_row = _load_gradient_kernels()
with torch.cuda.device(values.device):
    kernel[(values.shape[0], (values.shape[1] + CLASS_TILE_SIZE - 1) // CLASS_TILE_SIZE, (values.shape[2] + TIME_TILE_SIZE - 1) // TIME_TILE_SIZE)](
        values, targets, maxima, log_sums, row_scales, output,
        values.shape[2], ctx.full_shape[1], values.shape[1], ctx.ignore_index, reconstruct,
        class_tile=CLASS_TILE_SIZE, time_tile=TIME_TILE_SIZE, num_warps=4, enable_fp_fusion=True)
    zero_last_row[(values.shape[0], (values.shape[1] + 1023) // 1024)](
        output, ctx.full_shape[1], values.shape[1], block_size=1024, num_warps=4, enable_fp_fusion=False)
return output, None, None
'''),
        'input_contract': (functions.get('_validate_inputs'), '''
import torch
if (logits.ndim != 3 or targets.ndim != 2 or logits.shape[1] < 2
        or targets.shape != (logits.shape[0], logits.shape[1] - 1)):
    raise ValueError("Causal gradient CE requires full logits [batch,time,vocabulary] and shifted targets [batch,time-1]")
with torch.no_grad():
    _validate_spatial_inputs(logits[:, :-1, :].transpose(1, 2), targets, ignore_index)
'''),
        'public_dispatch': (functions.get('causal_gradient_token_cross_entropy'), '''
_validate_inputs(logits, targets, ignore_index)
return _autograd_function().apply(logits, targets, ignore_index)
'''),
    }
    for name, (function, source) in expected.items():
        body = [] if function is None else [node for node in function.body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str))]
        actual = [ast.dump(node, include_attributes=False) for node in body]
        wanted = [ast.dump(node, include_attributes=False) for node in ast.parse(source).body]
        if actual != wanted:
            raise ValueError('Causal gradient layout changed its protected '+name+' contract')
        bindings['causal_gradient_'+name] = content_hash(actual)
    imports = [ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.ImportFrom)]
    wanted_imports = ast.parse('''
from .scheduled_spatial_loss import _autograd_function as _scheduled_autograd_function
from .spatial_loss import SpatialLossUnsupported, _validate_inputs as _validate_spatial_inputs
''').body
    if any(ast.dump(node, include_attributes=False) not in imports for node in wanted_imports):
        raise ValueError('Causal gradient layout changed its accepted helper delegation')
    constants = {target.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant) for target in node.targets if isinstance(target, ast.Name)}
    if constants.get('CLASS_TILE_SIZE') != 32 or constants.get('TIME_TILE_SIZE') != 32:
        raise ValueError('Causal gradient layout changed its declared tiles')
    bindings['causal_gradient_loss_module'] = file_hash(path)
    return bindings


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
    if loss_implementation not in ('spatial_ce', 'flat_ce', 'fused_ce', 'spatial_fused_backward', 'spatial_scheduled_ce',
                                 'causal_contiguous_gradient_ce'):
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
        spatial_expression = ast.parse('token_losses = spatial_token_cross_entropy('
            'shifted_logits.transpose(1, 2), shifted_labels, ignore_index=-100)').body[0]
        spatial_contract = ast.dump(spatial_expression, include_attributes=False)
        scheduled_expression = ast.parse('token_losses = scheduled_spatial_token_cross_entropy('
            'shifted_logits.transpose(1, 2), shifted_labels, ignore_index=-100)').body[0]
        scheduled_contract = ast.dump(scheduled_expression, include_attributes=False)
        causal_expression = ast.parse('token_losses = causal_gradient_token_cross_entropy('
            'logits.float(), shifted_labels, ignore_index=-100)').body[0]
        causal_contract = ast.dump(causal_expression, include_attributes=False)
        dispatches = {'flat_ce': flat_contract, 'fused_ce': fused_contract,
                      'spatial_fused_backward': spatial_contract,
                      'spatial_scheduled_ce': scheduled_contract,
                      'causal_contiguous_gradient_ce': causal_contract}
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
                or any(value not in dispatches.values() for value in ce_dispatch)
                or dispatches[loss_implementation] not in ce_dispatch):
            raise ValueError('Replacement CE dispatch changed the frozen shift, mask or per-example weighting')
        bindings.update({'replacement_'+key: content_hash(value) for key,value in wrapper_contract.items()})
        bindings[loss_implementation+'_dispatch'] = content_hash(dispatches[loss_implementation])
        if loss_implementation == 'fused_ce':
            bindings['fused_ce_module'] = file_hash(ROOT/'src/training/sft/fused_loss.py')
        elif loss_implementation == 'spatial_fused_backward':
            bindings.update(spatial_native_forward_contract())
        elif loss_implementation == 'spatial_scheduled_ce':
            bindings.update(scheduled_spatial_forward_contract())
        elif loss_implementation == 'causal_contiguous_gradient_ce':
            bindings.update(causal_gradient_layout_contract())
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


def baseline_runtime_controls(path, config, commit, control_config=None):
    """Bind fresh native baseline controls without weakening optimization gates."""
    report = checked_report(path)
    measured_config = config if control_config is None else control_config
    if control_config is not None:
        from training.sft.data_handoff import compatibility_projection
        if compatibility_projection(config) != compatibility_projection(control_config):
            raise ValueError('Scientific settings changed after measured baseline controls')
    measured_commit = measured_config.get('runtime', {}).get('implementation_revision', commit)
    models = report.get('model_ids')
    if (report.get('schema_version') != 'sft_baseline_runtime_controls_v1'
            or report.get('status') != 'PASSED'
            or report.get('source_code_commit') != measured_commit
            or report.get('consumer_configuration_hash') != configuration_hash(measured_config)
            or report.get('execution_profile') != SFTExecutionProfile().to_dict()
            or report.get('dependency_hash') != file_hash(ROOT/'requirements.lock')
            or not isinstance(models, list) or len(models) != 1
            or models[0] not in config['models']
            or report.get('models') != {models[0]:config['models'][models[0]]}
            or report.get('training_started') is not False
            or report.get('optimization_activated') is not False):
        raise ValueError('Fresh baseline runtime control identity differs')
    if report.get('control_configuration_hash') != configuration_hash(measured_config):
        raise ValueError('Native controls changed the frozen production config')
    hashes = report.get('source_file_hashes', {})
    expected = {str(p.relative_to(ROOT)) for p in (ROOT/'src/training/sft').rglob('*.py')}
    expected.update(('src/shared_procedures/preflight.py','src/shared_procedures/checkpointing.py',
                     'requirements.lock','scripts/qualify_sft_baseline_runtime.py'))
    if set(hashes) != expected or any(file_hash(ROOT/name) != digest for name,digest in hashes.items()):
        raise ValueError('Measured native baseline source differs')
    checks = {}
    for name in ('model_runtime','loss_mask_gradients','full_checkpoint_resume'):
        row = report.get('checks', {}).get(name, {})
        evidence = row.get('evidence', {})
        if (row.get('name') != name or row.get('status') != 'passed'
                or evidence.get('live') is not True or evidence.get('simulation') is not False
                or evidence.get('control_preflight') is not True
                or evidence.get('scientific_results') is not False
                or not isinstance(evidence.get('models'), list) or len(evidence['models']) != 1):
            raise ValueError('Native baseline runtime check did not actually pass: '+name)
        measured = evidence['models'][0]
        if name == 'model_runtime':
            if measured.get('model') != config['models'][models[0]]:
                raise ValueError('Measured native model differs from the pinned construction model')
            if measured.get('runtime', {}).get('execution_profile') != SFTExecutionProfile().to_dict():
                raise ValueError('Measured runtime was not the published baseline')
        elif measured.get('model') != models[0]:
            raise ValueError('Measured native control model scope differs')
        copied = deepcopy(row)
        copied['evidence']['native_baseline_binding'] = {
            'report_hash':report['report_hash'], 'model_ids':models,
            'consumer_commit':commit, 'measured_source_commit':measured_commit,
            'measured_configuration_hash':configuration_hash(measured_config),
            'hardware':report['hardware']}
        checks[name] = CheckResult(**copied)
    return report, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','certificate','control-config','baseline-root','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--execution-benchmark')
    parser.add_argument('--execution-profile')
    parser.add_argument('--recovery-benchmark')
    parser.add_argument('--baseline-runtime-controls', help='Fresh model-scoped native baseline controls on replacement hardware.')
    parser.add_argument('--baseline-runtime-control-config', help='Original measured config for byte-identical native-runtime source reuse.')
    args = parser.parse_args(argv)
    if bool(args.execution_benchmark) != bool(args.execution_profile):
        parser.error('--execution-benchmark and --execution-profile must be supplied together')
    if args.recovery_benchmark and not args.execution_benchmark:
        parser.error('--recovery-benchmark requires the execution benchmark')
    if args.baseline_runtime_controls and args.execution_benchmark:
        parser.error('Native baseline controls cannot authorize an optimization profile')
    if args.baseline_runtime_control_config and not args.baseline_runtime_controls:
        parser.error('Original baseline runtime config requires the measured controls')
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
    native_baseline = None
    if args.baseline_runtime_controls:
        native_baseline, execution_checks = baseline_runtime_controls(args.baseline_runtime_controls, config, commit,
            read_json(args.baseline_runtime_control_config) if args.baseline_runtime_control_config else None)
    functions = (['render_sft_completion','tokenize_completion','capture_rng','restore_rng','exposure_order',
                  'SFTRecipe','TARGET_MODULES','LOSS_REDUCTION','MODELS','LEVELS'] if accepted or native_baseline else
                 ['_runtime','render_sft_completion','tokenize_completion','verify_mask_gradients','collate_completion',
                  'completion_loss','capture_rng','restore_rng','train_optimizer_updates','SFTRecipe',
                  'TARGET_MODULES','LOSS_REDUCTION','MODELS','LEVELS'])
    bindings = {'gpu_functions':compatible_functions(gpu_source,
        'src/training/sft/individual_fit/sft_fit.py',
        functions),
        'gpu_control':({'native_source_sha256':native_baseline['source_file_hashes']['src/shared_procedures/preflight.py']}
                       if native_baseline else compatible_functions(gpu_source,'src/shared_procedures/preflight.py',['_run_model_checks'])),
        'checkpoint':identical_files(gpu_source,['src/shared_procedures/checkpointing.py',
            'requirements.lock','src/operations/assets.py','src/shared_procedures/identity.py',
            'src/shared_procedures/response_parsing.py','src/shared_procedures/storage.py'])}
    if accepted or native_baseline:
        bindings['scientific_sft_numerics'] = compatible_sft_numerics(gpu_source,
            loss_implementation=accepted['execution_profile'].get('loss_implementation', 'spatial_ce') if accepted else 'spatial_ce')
        if accepted:
            bindings['accepted_execution'] = accepted
        else:
            bindings['native_baseline'] = {'report_hash':native_baseline['report_hash'],
                                         'model_ids':native_baseline['model_ids']}
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
    if native_baseline:
        report['ready_models'] = native_baseline['model_ids']
        report['baseline_only'] = True
        report['study_ready'] = False
    if accepted:
        report['ready_scope'] = accepted['condition']
        report['study_ready'] = False
        report['entire_model_condition_matrix_retested'] = False
    if accepted or native_baseline:
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
    if native_baseline:
        receipt.update(ready_models=native_baseline['model_ids'], baseline_only=True,
            study_ready=False, native_baseline_report_hash=native_baseline['report_hash'],
            target_sft_controls_repeated=True, entire_model_condition_matrix_retested=False)
    receipt['receipt_hash'] = content_hash(receipt)
    atomic_write_json(Path(args.output)/'readiness_receipt.json',receipt)
    print(__import__('json').dumps({k:v for k,v in receipt.items() if k != 'source_compatibility_bindings'}),flush=True)


if __name__ == '__main__':
    main()
