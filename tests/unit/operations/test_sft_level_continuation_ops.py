"""Restore trust boundaries and serial recovery orchestration without GPU/network."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations.artifacts import ArtifactService
from shared_procedures.checkpointing import save_checkpoint
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, read_json
from training.sft.individual_fit.sft_fit import TARGET_MODULES
from training.sft.level_continuation import ContinuationSpec

ROOT = Path(__file__).resolve().parents[3]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RESTORE = script("restore_sft_level_continuation_inputs")
CAMPAIGN = script("run_sft_level_continuation_campaign")


def sealed(value, field):
    value[field] = content_hash({key: item for key, item in value.items() if key != field})
    return value


def parent_fixture(tmp_path):
    root = tmp_path / "source"
    paths = RESTORE.level_paths(1)
    identity = {"family": "rot13", "model_id": "14b", "level": 1, "run_id": "rot13_14b_level_01",
                "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                "model_revision": RESTORE.MODEL_REVISION, "tokenizer_revision": RESTORE.MODEL_REVISION,
                "configuration_hash": "a" * 64}
    progress = {"completed_steps": 625, "total_optimizer_steps": 625, "sampler_position": 10000,
                "example_presentations": 10000, "physical_microbatch": 1, "gradient_accumulation": 16,
                "gradient_checkpointing": False}
    checkpoint = save_checkpoint(root / paths["parent_checkpoint"], identity, {
        "adapter": {"weight": 1}, "optimizer": {"state": {1: "initialized"}},
        "scheduler": {"last_epoch": 625}, "rng": {"python": "retained"}, "progress": progress})
    adapter = root / paths["parent_fit"] / "final_adapter"
    atomic_write_json(adapter / "adapter_config.json", {"r": 8, "lora_alpha": 16,
        "lora_dropout": 0.05, "target_modules": list(TARGET_MODULES),
        "base_model_name_or_path": identity["model_name"]})
    (adapter / "adapter_model.safetensors").write_bytes(b"checksummed adapter fixture")
    adapter_manifest = sealed({"identity": identity, "checkpoint_hash": checkpoint["checkpoint_hash"],
        "files": {p.name: file_hash(p) for p in adapter.iterdir()}}, "manifest_hash")
    atomic_write_json(adapter / "manifest.json", adapter_manifest)
    atomic_write_json(root / paths["parent_fit"] / "fit_result.json", {
        "status": "complete", "identity": identity, "progress": {**progress, "status": "complete"},
        "checkpoint_hash": checkpoint["checkpoint_hash"], "adapter_manifest_hash": adapter_manifest["manifest_hash"],
        "adapter_files": adapter_manifest["files"]})
    for key in ("train", "validation"):
        path = root / paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"fixture": true}\n')
    atomic_write_json(root / paths["train"].replace("train.jsonl", "manifest.json"), {
        "status": "verified", "simulation": False, "family": "rot13", "model_id": "14b", "level": 1,
        "train_count": 10000, "validation_count": 100})
    atomic_write_json(root / RESTORE.DATA_ROOT / "manifest.json", {"fixture": "audited bundle"})
    records = []
    for index, path in enumerate(sorted(p for p in root.rglob("*") if p.is_file() and p.suffix != ".lock")):
        records.append({"relative_path": str(path.relative_to(root)), "artifact_id": content_hash([index, path.name]),
            "configuration_hash": "a" * 64, "sha256": file_hash(path), "size": path.stat().st_size,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(), "drive_file_id": f"file{index}",
            "drive_folder_id": "original-parent-folder"})
    entry = {"level": 1, **paths, "parent_identity_hash": content_hash(identity),
             "parent_checkpoint_hash": checkpoint["checkpoint_hash"]}
    inventory = sealed({"version": RESTORE.VERSION, "project_folder_id": RESTORE.PROJECT_FOLDER_ID,
        "model_revision": RESTORE.MODEL_REVISION, "family": "rot13", "model": "14b",
        "levels": [entry], "artifacts": records}, "inventory_hash")
    return root, inventory


def test_restore_verifies_complete_parent_before_publishing_inputs_manifest(tmp_path):
    root, inventory = parent_fixture(tmp_path)
    destination = tmp_path / "restored"
    events = []
    by_id = {record["artifact_id"]: record for record in inventory["artifacts"]}

    class Store:
        folder_id = RESTORE.PROJECT_FOLDER_ID
        def create_project_folder(self):
            return self.folder_id
        def verify_registered_artifact(self, record):
            events.append(("verify", record["artifact_id"]))
        def resolve_artifact_id(self, artifact_id):
            return {"id": by_id[artifact_id]["drive_file_id"]}
        def download_artifact(self, artifact_id, path, *, config_hash):
            record = by_id[artifact_id]
            assert config_hash == record["configuration_hash"]
            path.write_bytes((root / record["relative_path"]).read_bytes())
            return path

    result = RESTORE.restore_inventory(inventory, destination, store_factory=Store, workers=2)
    assert result["status"] == "VERIFIED" and result["downloaded"] == len(by_id)
    manifest = read_json(result["inputs_manifest"])
    entries = RESTORE.validate_inputs_manifest(manifest)
    assert entries[0]["parent_checkpoint"] == str(destination / inventory["levels"][0]["parent_checkpoint"])
    assert len(events) >= len(by_id) * 3
    assert RESTORE.restore_inventory(inventory, destination, validate_only=True)["status"] == "VALIDATED"
    payload = destination / inventory["levels"][0]["parent_checkpoint"] / "full_state.pt"
    payload.write_bytes(payload.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity"):
        RESTORE.validate_inputs_manifest(manifest)


@pytest.mark.parametrize("mutation", ["pilot_path", "missing_state", "other_level", "duplicate", "unsealed"])
def test_restore_rejects_unsafe_or_incomplete_inventory_before_transfer(tmp_path, mutation):
    _, inventory = parent_fixture(tmp_path)
    if mutation == "pilot_path":
        inventory["artifacts"][0]["relative_path"] = "training/sft/rot13/14b/level_10_continuation/full_state.pt"
    elif mutation == "missing_state":
        inventory["artifacts"] = [record for record in inventory["artifacts"] if not record["relative_path"].endswith("full_state.pt")]
    elif mutation == "other_level":
        inventory["levels"][0]["level"] = 2
    elif mutation == "duplicate":
        inventory["artifacts"].append(deepcopy(inventory["artifacts"][0]))
    if mutation != "unsealed":
        sealed(inventory, "inventory_hash")
    else:
        inventory["family"] = "kazitext"
    with pytest.raises(ValueError):
        RESTORE.restore_inventory(inventory, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()


def test_restore_rejects_symbolic_link_and_preserves_existing_different_bytes(tmp_path):
    root, inventory = parent_fixture(tmp_path)
    destination = tmp_path / "restore"
    destination.mkdir()
    (destination / "training").symlink_to(root / "training", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        RESTORE.restore_inventory(inventory, destination)
    assert (destination / "training").is_symlink()


def test_campaign_shard_is_explicit_and_invokes_existing_single_level_worker(tmp_path):
    _, inventory = parent_fixture(tmp_path)
    entries = inventory["levels"]
    manifest = {"manifest_hash": "f" * 64}
    plan = CAMPAIGN.campaign_plan(manifest, entries, [1], source="b" * 40,
                                 additional_epochs=1, learning_rate=5e-5)
    job = plan["jobs"][0]
    assert job["policy"]["cumulative_final_updates"] == 1250
    assert job["policy"]["learning_rate_peak"] == 5e-5
    args = SimpleNamespace(python="gpu/bin/python", model_cache="models", additional_epochs=1,
                           learning_rate=5e-5, device="cuda:0")
    command = CAMPAIGN.training_command(job, args, tmp_path / "new-output")
    assert command[2].endswith("continue_sft_level.py")
    assert command[command.index("--parent-checkpoint") + 1].endswith("checkpoint_000625")
    assert command[command.index("--level") + 1] == "1"
    for levels in ([1, 1], [2]):
        with pytest.raises(ValueError):
            CAMPAIGN.campaign_plan(manifest, entries, levels, source="b" * 40,
                                   additional_epochs=1, learning_rate=5e-5)


def completion_fixture(tmp_path, monkeypatch):
    output = tmp_path / "job"
    spec = ContinuationSpec(1, 1, 5e-5)
    identity = {"run_id": "fixture_v2", "configuration_hash": "b" * 64, "code_revision": "c" * 40,
        "continuation_policy": spec.policy(), "parent": {"checkpoint_hash": "a" * 64, "identity_hash": "d" * 64}}
    job = {"level": 1, "parent_checkpoint_hash": "a" * 64, "parent_identity_hash": "d" * 64,
           "policy": spec.policy()}
    result = {"identity": identity, "checkpoint_hash": "e" * 64, "adapter_manifest_hash": "f" * 64,
              "progress": {"completed_steps": 625, "cumulative_completed_steps": 1250}}
    atomic_write_json(output / "run_identity.json", identity)
    atomic_write_json(output / "fit_result.json", result)
    for name in ("checkpoint_000625/full_state.pt", "checkpoint_000625/manifest.json", "final_adapter/adapter_model.safetensors"):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"final payload " + name.encode())
    service = ArtifactService(output, output / "drive_registry.json")
    provenance = {"experiment_id": "fixture", "branch": "continuation", "model": "14b", "run_id": identity["run_id"],
        "stage": "sft", "producer": "fixture", "source_code_commit": "c" * 40,
        "configuration_hash": "b" * 64, "dataset_revision": "d" * 64,
        "model_revision": "a" * 40, "seeds": [42]}
    records = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.suffix not in (".sqlite3", ".lock")):
        if "drive_registry" in path.name:
            continue
        record = service.register(path, provenance)
        record.update(sync_status="VERIFIED", drive_file_id="remote-" + path.name, drive_folder_id="project")
        records[record["artifact_id"]] = record
    payload = {"version": "sft_level_continuation_drive_payloads_v2", "configuration_hash": "b" * 64, "artifacts": deepcopy(records)}
    atomic_write_json(output / "drive_payload_manifest.json", payload)
    receipt = {"version": "sft_level_continuation_completion_receipt_v2", "status": "COMPLETE", "drive_status": "VERIFIED",
        "simulation": False, "run_id": identity["run_id"], "configuration_hash": "b" * 64, "source_code_commit": "c" * 40,
        "identity_hash": content_hash(identity), "parent_checkpoint_hash": "a" * 64, "continuation_policy": spec.policy(),
        "fit_result_sha256": file_hash(output / "fit_result.json"), "checkpoint_hash": "e" * 64,
        "adapter_manifest_hash": "f" * 64, "progress": result["progress"],
        "payload_manifest_sha256": file_hash(output / "drive_payload_manifest.json")}
    atomic_write_json(output / "completion_receipt.json", sealed(receipt, "receipt_hash"))
    for name in ("drive_payload_manifest.json", "completion_receipt.json"):
        record = service.register(output / name, provenance)
        record.update(sync_status="VERIFIED", drive_file_id="remote-" + name, drive_folder_id="project")
        records[record["artifact_id"]] = record
    monkeypatch.setattr(CAMPAIGN, "verified_completion", lambda out, ident: result)
    calls = []
    store = SimpleNamespace(verify_registered_artifact=lambda record: calls.append(record["relative_path"]))
    return output, job, records, store, calls


def test_complete_flag_needs_full_registry_and_live_drive_verification(tmp_path, monkeypatch):
    output, job, records, store, calls = completion_fixture(tmp_path, monkeypatch)
    # Receipt is durable locally before the final registry export: recovery must
    # re-enter the worker's idempotent publisher, never fail or silently skip.
    assert CAMPAIGN.verified_job_completion(output, job, "c" * 40, store=store) is None
    atomic_write_json(output / "drive_registry.json", {"version": 2, "artifacts": records})
    endpoint = CAMPAIGN.verified_job_completion(output, job, "c" * 40, store=store)
    assert endpoint["drive_status"] == "VERIFIED"
    assert len(calls) == len(records)
    records[next(iter(records))]["sync_status"] = "PENDING"
    atomic_write_json(output / "drive_registry.json", {"version": 2, "artifacts": records})
    assert CAMPAIGN.verified_job_completion(output, job, "c" * 40, store=store) is None
    job["parent_checkpoint_hash"] = "0" * 64
    with pytest.raises(ValueError, match="exact condition"):
        CAMPAIGN.verified_job_completion(output, job, "c" * 40, store=store)


def campaign_fixture(tmp_path, monkeypatch):
    entries = [{"level": level, **RESTORE.level_paths(level), "parent_checkpoint_hash": str(level) * 64,
                "parent_identity_hash": "a" * 64} for level in (1, 2)]
    manifest = {"manifest_hash": "d" * 64}
    path = tmp_path / "inputs.json"
    atomic_write_json(path, manifest)
    cache = tmp_path / "models"
    cache.mkdir()
    monkeypatch.setattr(CAMPAIGN, "validate_inputs_manifest", lambda value: entries)
    monkeypatch.setattr(CAMPAIGN, "code_revision", lambda: "c" * 40)
    args = SimpleNamespace(inputs_manifest=str(path), output=str(tmp_path / "campaign"), levels=[1, 2],
        model_cache=str(cache), additional_epochs=1, learning_rate=5e-5, python="gpu/python",
        device="cuda:0", plan_only=False)
    return args


def test_failed_child_prevents_later_level_launch(tmp_path, monkeypatch):
    args = campaign_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(CAMPAIGN, "verified_job_completion", lambda *args, **kwargs: None)
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(poll=lambda: 2, returncode=2)
    with pytest.raises(RuntimeError, match="later levels"):
        CAMPAIGN.run_campaign(args, popen=popen, store=object())
    assert len(commands) == 1
    assert commands[0][commands[0].index("--level") + 1] == "1"
    assert read_json(Path(args.output) / "campaign_status.json")["status"] == "FAILED"


def test_child_pause_stops_shard_without_next_level_and_plan_only_writes_nothing(tmp_path, monkeypatch):
    args = campaign_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(CAMPAIGN, "verified_job_completion", lambda *args, **kwargs: None)
    calls = []
    def popen(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(poll=lambda: 3, returncode=3)
    result = CAMPAIGN.run_campaign(args, popen=popen, store=object())
    assert result["status"] == "PAUSED" and len(calls) == 1
    args.output = str(tmp_path / "plan_only_must_not_exist")
    args.plan_only = True
    assert CAMPAIGN.run_campaign(args)["status"] == "PLANNED"
    assert not Path(args.output).exists()


def test_checkpointing_campaign_has_distinct_identity_and_passes_flag(tmp_path):
    _, inventory = parent_fixture(tmp_path)
    kw = dict(source='b' * 40, additional_epochs=1, learning_rate=5e-5)
    regular = CAMPAIGN.campaign_plan({'manifest_hash': 'f' * 64}, inventory['levels'], [1], **kw)
    selected = CAMPAIGN.campaign_plan({'manifest_hash': 'f' * 64}, inventory['levels'], [1],
                                      enable_gradient_checkpointing=True, **kw)
    assert regular['plan_hash'] != selected['plan_hash']
    assert regular['jobs'][0]['job_hash'] != selected['jobs'][0]['job_hash']
    args = SimpleNamespace(python='python', model_cache='models', additional_epochs=1,
                           learning_rate=5e-5, device='cuda:0', enable_gradient_checkpointing=True)
    assert '--enable-gradient-checkpointing' in CAMPAIGN.training_command(selected['jobs'][0], args, tmp_path/'out')
