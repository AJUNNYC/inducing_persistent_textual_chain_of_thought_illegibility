"""Focused CPU checks for matched-level extra exposure and exact recovery."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from shared_procedures.checkpointing import _same_state, save_checkpoint
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json
from training.sft import continuation as frozen
from training.sft import level_continuation as c
from training.sft.individual_fit.sft_fit import (
    SFTFitSpec, SFTInputError, TARGET_MODULES, dataset_hash, exposure_order,
)
from training.sft.published_execution import PUBLISHED_EXECUTION_POLICY

ROOT = Path(__file__).resolve().parents[4]
CLI_SPEC = importlib.util.spec_from_file_location("continue_sft_level_test_cli", ROOT / "scripts/continue_sft_level.py")
CLI = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(CLI)


def runtime(spec=None, parent_hash="a" * 64):
    result = {"physical_microbatch": 1, "gradient_accumulation": 16,
              "gradient_checkpointing": False, "published_execution_policy": PUBLISHED_EXECUTION_POLICY}
    if spec:
        result.update(continuation_policy=spec.policy(), parent_checkpoint_hash=parent_hash)
    return result


def identity(spec):
    return {"run_id": "fixture_extra", "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "configuration_hash": "b" * 64, "code_revision": "c" * 40,
            "continuation_policy": spec.policy(),
            "parent": {"checkpoint_hash": "a" * 64},
            "exposure_order_hash": content_hash(c.added_exposure_order(spec))}


def initialized_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4, weight_decay=0)
    for _ in range(4):
        parameter.grad = torch.tensor([0.3, -0.1])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    optimizer.param_groups[0]["lr"] = 0.0
    return parameter, optimizer


@pytest.mark.parametrize("epochs", [1, 2])
def test_real_adamw_resume_preserves_exact_next_update(epochs):
    spec = c.ContinuationSpec(4, epochs, 5e-5)
    parameter, optimizer = initialized_optimizer()
    original = deepcopy(optimizer.state_dict()["state"])
    scheduler = c.restore_optimizer_scheduler(optimizer, {"optimizer": deepcopy(optimizer.state_dict())},
                                               spec, continuing_phase=False)
    assert _same_state(optimizer.state_dict()["state"], original)
    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5 / spec.warmup_steps)
    for _ in range(23):
        parameter.grad = torch.tensor([0.2, -0.4])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    state = {"optimizer": deepcopy(optimizer.state_dict()), "scheduler": deepcopy(scheduler.state_dict()),
             "progress": {"completed_steps": 23}}
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = torch.optim.AdamW([restored_parameter], lr=0.9)
    restored_scheduler = c.restore_optimizer_scheduler(restored, state, spec, continuing_phase=True)
    for p, opt, sched in ((parameter, optimizer, scheduler), (restored_parameter, restored, restored_scheduler)):
        p.grad = torch.tensor([0.6, -0.7])
        opt.step()
        sched.step()
    assert torch.equal(parameter, restored_parameter)
    assert _same_state(optimizer.state_dict(), restored.state_dict())
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    assert c.learning_rate_scale(spec.total_steps, spec) == 0
    assert all(0 < c.learning_rate_scale(step, spec) <= 1 for step in range(spec.total_steps))


def test_wrong_lr_cursor_and_invalid_policy_are_rejected():
    spec = c.ContinuationSpec(10, 1, 5e-5)
    _, optimizer = initialized_optimizer()
    scheduler = c.restore_optimizer_scheduler(optimizer, {"optimizer": deepcopy(optimizer.state_dict())},
                                               spec, continuing_phase=False)
    state = {"optimizer": deepcopy(optimizer.state_dict()), "scheduler": deepcopy(scheduler.state_dict()),
             "progress": {"completed_steps": 4}}
    with pytest.raises(SFTInputError, match="scheduler/LR"):
        c.restore_optimizer_scheduler(optimizer, state, spec, continuing_phase=True)
    for arguments in ((0, 1, 5e-5), (11, 1, 5e-5), (True, 1, 5e-5),
                      (1, 0, 5e-5), (1, 3, 5e-5), (1, 1, 0), (1, 1, float("nan")), (1, 1, True)):
        with pytest.raises(SFTInputError):
            c.ContinuationSpec(*arguments)


def test_parent_namespace_is_rejected_before_any_output_is_created(tmp_path, monkeypatch):
    parent = tmp_path / "original"
    checkpoint = parent / "checkpoint_000625"
    for output in (parent, parent / "new_fit", checkpoint, tmp_path):
        with pytest.raises(SFTInputError, match="separate from immutable"):
            c.validate_output_directory(output, parent, checkpoint)
    assert not parent.exists()
    assert c.validate_output_directory(tmp_path / "new_phase", parent, checkpoint) == tmp_path / "new_phase"
    arguments = ["continue_sft_level", "--parent-fit", str(parent), "--parent-checkpoint", str(checkpoint),
        "--train", "unused", "--validation", "unused", "--output", str(parent), "--model-cache", "unused",
        "--level", "1", "--additional-epochs", "1", "--learning-rate", "5e-5"]
    monkeypatch.setattr("sys.argv", arguments)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Overlapping output must be rejected before setup or corpus loading")
    monkeypatch.setattr(CLI, "configure_worker_recovery", forbidden)
    monkeypatch.setattr(CLI, "load_sft_records", forbidden)
    with pytest.raises(SFTInputError, match="separate from immutable"):
        CLI.main()
    assert not parent.exists()


def test_phase_order_preserves_second_epoch_and_frozen_v1_policy():
    one = c.ContinuationSpec(10, 1, 5e-5)
    two = c.ContinuationSpec(10, 2, 5e-5)
    assert c.added_exposure_order(one) == exposure_order(10000, 42, epochs=2)[10000:]
    assert c.added_exposure_order(two)[:10000] == c.added_exposure_order(one)
    assert sorted(c.added_exposure_order(two)[10000:]) == list(range(10000))
    assert c.added_exposure_order(two)[10000:] != c.added_exposure_order(one)
    assert {313, 625} <= c.checkpoint_steps(one)
    assert {313, 625, 938, 1250} <= c.checkpoint_steps(two)
    assert frozen.POLICY["learning_rate_peak"] == 2e-4
    assert frozen.POLICY["additional_epochs"] == 1
    assert frozen.POLICY["condition"]["level"] == 10
    with pytest.raises(SFTInputError, match="one- or two-epoch"):
        exposure_order(10000, 42, epochs=3)


def test_pause_resume_crosses_epoch_boundary_without_replaying_examples(monkeypatch):
    spec = c.ContinuationSpec(2, 2, 5e-5)
    ident, mode = identity(spec), runtime(spec)
    examples = [{"example_id": index, "input_ids": [index]} for index in range(10000)]
    scheduler = SimpleNamespace(last_epoch=0)
    seen, saved = [], []
    def update(_model, _optimizer, sched, group, *_args, **_kwargs):
        seen.extend(example["example_id"] for example in group)
        sched.last_epoch += 1
    monkeypatch.setattr(c, "one_optimizer_update", update)
    model = SimpleNamespace(train=lambda: None)
    result = c.train_added_epochs(model, None, scheduler, examples, 0, identity=ident,
        runtime=mode, recipe=None, save=lambda cursor: saved.append(deepcopy(cursor)),
        should_pause=lambda: scheduler.last_epoch >= 624)
    assert result["status"] == "paused"
    assert len(seen) == 9984
    assert 313 in [cursor["completed_steps"] for cursor in saved]
    for boundary in (625, 626):
        result = c.train_added_epochs(model, None, scheduler, examples, 0, identity=ident,
            runtime=mode, recipe=None, progress=saved[-1], save=lambda cursor: saved.append(deepcopy(cursor)),
            should_pause=lambda: scheduler.last_epoch >= boundary)
    assert seen == c.added_exposure_order(spec)[:626 * 16]
    assert result["cumulative_completed_steps"] == 1251
    assert result["epoch_index"] == 3
    assert result["completed_additional_epochs"] == 1
    assert saved[-2]["completed_steps"] == 625


def original_parent(tmp_path, level=4):
    from safetensors.torch import save_file
    train, validation = [], []
    for split, count, target in (("train", 10000, train), ("validation", 100, validation)):
        for index in range(count):
            prompt = f"Question {split} {index}"
            target.append({"status": "verified", "split": split, "method": "rot13", "level": level,
                "model_id": "14b", "example_id": f"{split}_{index}", "prompt": prompt,
                "reasoning": "Ny cun", "final_answer": "B", "completion": "<think>Ny cun</think>B",
                "clean_trace_hash": "0" * 64, "source_content_hash": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                "source_manifest_hash": "e" * 64, "model_revision": "a" * 40,
                "tokenizer_revision": "b" * 40, "config_hash": "f" * 64, "seed": 42})
    spec = SFTFitSpec("rot13", "14b", level, "a" * 40, "b" * 40, "c" * 64,
        dataset_hash(train, validation), producer_configuration_hash="f" * 64,
        handoff_certificate_hash="0" * 64, training_budget=c.EFFICIENT_TRAINING_BUDGET)
    ident = {**spec.identity(), "code_revision": "d" * 40}
    mode = runtime()
    progress = {"completed_steps": 625, "total_optimizer_steps": 625, "sampler_position": 10000,
        "example_presentations": 10000, "exposure_order_hash": content_hash(exposure_order(10000, 42, epochs=1)),
        "execution_profile": None, **mode}
    weights = {"lora_test": torch.ones(2)}
    _, optimizer = initialized_optimizer()
    parent = tmp_path / "parent"
    checkpoint = save_checkpoint(parent / "checkpoint_000625", ident,
        {"adapter": weights, "optimizer": optimizer.state_dict(), "scheduler": {},
         "rng": {"fixture": "rng"}, "progress": progress, "runtime": mode})
    adapter = parent / "final_adapter"
    adapter.mkdir()
    save_file(weights, str(adapter / "adapter_model.safetensors"))
    atomic_write_json(adapter / "adapter_config.json", {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
        "target_modules": list(TARGET_MODULES), "base_model_name_or_path": ident["model_name"]})
    files = {path.name: file_hash(path) for path in adapter.iterdir()}
    manifest = {"identity": ident, "files": files, "checkpoint_hash": checkpoint["checkpoint_hash"]}
    manifest["manifest_hash"] = content_hash(manifest)
    atomic_write_json(adapter / "manifest.json", manifest)
    result = {"status": "complete", "identity": ident, "progress": {**progress, "status": "complete"},
        "runtime": mode, "checkpoint_hash": checkpoint["checkpoint_hash"],
        "adapter_manifest_hash": manifest["manifest_hash"], "adapter_files": files,
        "validation_count": 100, "validation_loss": 0.7}
    atomic_write_json(parent / "fit_result.json", result)
    return parent, parent / "checkpoint_000625", train, validation


def test_input_binding_rejects_other_level_changed_corpus_and_pilot(tmp_path, monkeypatch):
    parent, checkpoint, train, validation = original_parent(tmp_path)
    monkeypatch.setattr(c, "_tokenized_data", lambda *args: ({"fixture": "same_native_renderer"}, [], []))
    continuation = c.ContinuationSpec(4, 1, 5e-5)
    ident, _, _, _ = c.prepare_inputs(parent, checkpoint, train, validation, "d" * 40,
                                    continuation=continuation, cache_root=tmp_path)
    assert ident["parent"]["identity"]["run_id"] == "rot13_14b_level_04"
    assert ident["continuation_policy"]["learning_rate_peak"] == 5e-5
    with pytest.raises(SFTInputError, match="original published"):
        c.prepare_inputs(parent, checkpoint, train, validation, "d" * 40,
                         continuation=c.ContinuationSpec(5, 1, 5e-5), cache_root=tmp_path)
    changed = deepcopy(train)
    changed[0]["reasoning"] += " changed"
    changed[0]["completion"] = "<think>" + changed[0]["reasoning"] + "</think>B"
    with pytest.raises(SFTInputError, match="record bytes"):
        c.prepare_inputs(parent, checkpoint, changed, validation, "d" * 40,
                         continuation=continuation, cache_root=tmp_path)
    result = c.read_json(parent / "fit_result.json")
    result["identity"]["continuation_policy"] = deepcopy(frozen.POLICY)
    atomic_write_json(parent / "fit_result.json", result)
    with pytest.raises(SFTInputError, match="original published"):
        c.prepare_inputs(parent, checkpoint, train, validation, "d" * 40,
                         continuation=continuation, cache_root=tmp_path)


@pytest.mark.parametrize("epochs", [1, 2])
def test_completed_republication_is_immutable_and_never_initializes_gpu(tmp_path, monkeypatch, epochs):
    from safetensors.torch import save_file
    spec = c.ContinuationSpec(3, epochs, 5e-5)
    ident, mode = identity(spec), runtime(spec)
    output = tmp_path / "continued"
    progress = c.progress_record(spec.total_steps, 20000, ident, mode)
    weights = {"lora_test": torch.ones(2)}
    checkpoint = save_checkpoint(c.final_checkpoint(output, ident), ident,
        {"adapter": weights, "optimizer": {"fixture": "moments"}, "scheduler": {},
         "rng": {"fixture": "rng"}, "progress": progress, "runtime": mode})
    adapter = output / "final_adapter"
    adapter.mkdir()
    save_file(weights, str(adapter / "adapter_model.safetensors"))
    atomic_write_json(adapter / "adapter_config.json", {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
        "target_modules": list(TARGET_MODULES), "base_model_name_or_path": ident["model_name"]})
    files = {path.name: file_hash(path) for path in adapter.iterdir()}
    manifest = {"identity": ident, "files": files, "checkpoint_hash": checkpoint["checkpoint_hash"]}
    manifest["manifest_hash"] = content_hash(manifest)
    atomic_write_json(adapter / "manifest.json", manifest)
    result = {"status": "complete", "identity": ident, "progress": {**progress, "status": "complete"},
        "checkpoint_hash": checkpoint["checkpoint_hash"], "adapter_manifest_hash": manifest["manifest_hash"],
        "adapter_files": files, "validation_count": 100, "validation_loss": 0.7}
    atomic_write_json(output / "fit_result.json", result)
    before = (output / "fit_result.json").read_bytes()
    monkeypatch.setattr(c, "prepare_inputs", lambda *args, **kwargs: (ident, {}, [], []))
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed re-publication must not initialize a GPU")
    monkeypatch.setattr(c, "_runtime", forbidden)
    observed = c.run_continuation(tmp_path / "parent", tmp_path / "parent_checkpoint", [], [], output,
        source_revision="c" * 40, continuation=spec, cache_root=tmp_path)
    assert observed == result
    assert (output / "fit_result.json").read_bytes() == before
    assert c.final_checkpoint(output, ident).name == f"checkpoint_{625 * epochs:06d}"


def test_drive_barrier_rejects_unverified_final_payload(tmp_path):
    from operations.artifacts import ArtifactService
    spec, output = c.ContinuationSpec(2, 1, 5e-5), tmp_path / "continuation"
    ident = identity(spec)
    files = [output / "run_identity.json", output / "fit_result.json",
             c.final_checkpoint(output, ident) / "manifest.json",
             c.final_checkpoint(output, ident) / "full_state.pt",
             output / "final_adapter" / "manifest.json"]
    service = ArtifactService(output, output / "registry.json")
    provenance = {"experiment_id": "test", "branch": "rot13_level_continuation", "model": "14b",
        "run_id": ident["run_id"], "stage": "sft_level_continuation", "producer": "test",
        "source_code_commit": ident["code_revision"], "configuration_hash": ident["configuration_hash"],
        "dataset_revision": "d" * 64, "model_revision": "e" * 40, "seeds": [42]}
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
        record = service.register(path, provenance)
        record.update(sync_status="VERIFIED", drive_file_id="remote", drive_folder_id="folder")
        with service._connect() as database:
            service._put(database, record)
    assert len(CLI.verified_drive_records(service, output, ident)) == len(files)
    broken = service.list_artifacts()[0]
    broken["sync_status"] = "PENDING"
    with service._connect() as database:
        service._put(database, broken)
    with pytest.raises(RuntimeError, match="verified Drive"):
        CLI.verified_drive_records(service, output, ident)


def test_sealed_drive_receipt_retries_without_rewriting_scientific_payloads(tmp_path, monkeypatch):
    from operations.artifacts import ArtifactService
    from shared_procedures.storage import artifact_hook
    class MemoryDrive:
        def create_project_folder(self):
            return "project"
        def ensure_path(self, _parts):
            return "folder"
        def upload_artifact(self, path, *, artifact_id, folder_id, config_hash):
            return {"id": artifact_id}
    spec, output = c.ContinuationSpec(1, 1, 5e-5), tmp_path / "continued"
    ident = identity(spec)
    service = ArtifactService(output, output / "registry.json", MemoryDrive())
    provenance = {"experiment_id": "test", "branch": "rot13_level_continuation", "model": "14b",
        "run_id": ident["run_id"], "stage": "sft_level_continuation", "producer": "test",
        "source_code_commit": ident["code_revision"], "configuration_hash": ident["configuration_hash"],
        "dataset_revision": "d" * 64, "model_revision": "e" * 40, "seeds": [42]}
    result = {"checkpoint_hash": "f" * 64, "adapter_manifest_hash": "0" * 64,
              "progress": {"completed_steps": 625, "cumulative_completed_steps": 1250}}
    monkeypatch.setattr(CLI, "verified_completion", lambda *_args: result)
    paths = [output / "run_identity.json", output / "fit_result.json",
             c.final_checkpoint(output, ident) / "manifest.json",
             c.final_checkpoint(output, ident) / "full_state.pt",
             output / "final_adapter" / "manifest.json"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixed scientific bytes")
        service.register(path, provenance)
    with artifact_hook(service.completion_hook(provenance)):
        receipt = CLI.publish_drive_completion(service, output, ident)
        saved = {path: path.read_bytes() for path in paths +
                 [output / "drive_payload_manifest.json", output / "completion_receipt.json"]}
        assert receipt["drive_status"] == "VERIFIED"
        assert receipt["receipt_hash"] == content_hash({key: value for key, value in receipt.items()
                                                       if key != "receipt_hash"})
        assert CLI.publish_drive_completion(service, output, ident) == receipt
        assert all(path.read_bytes() == contents for path, contents in saved.items())


def test_checkpointing_is_declared_new_phase_only_and_preserves_training_budget():
    original = c.ContinuationSpec(5, 1, 5e-5)
    selected = c.ContinuationSpec(5, 1, 5e-5, True)
    assert {k: v for k, v in selected.policy().items() if k != 'physical_execution'} == {
        k: v for k, v in original.policy().items() if k != 'physical_execution'}
    assert c.spec_from_identity({'continuation_policy': selected.policy()}) == selected
    assert c.spec_from_identity({'continuation_policy': original.policy()}) == original
    assert c.continuation_checkpointing(selected, False, resuming=False) is True
    assert c.continuation_checkpointing(selected, True, resuming=True) is True
    assert c.continuation_checkpointing(original, False, resuming=True) is False
    with pytest.raises(c.SFTInputError, match='declared checkpointing'):
        c.continuation_checkpointing(selected, False, resuming=True)
