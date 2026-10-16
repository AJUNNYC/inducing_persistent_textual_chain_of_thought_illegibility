"""CPU operations checks: immutable scientific bindings and a one-state detour."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, file_lock, read_json

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("gpqa_priority", ROOT / "scripts/run_gpqa_frozen_priority_campaign.py")
PRIORITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRIORITY)


def seal(record, field):
    record[field] = content_hash(record)
    return record


def sealed(record, field):
    if record[field] != content_hash({k: v for k, v in record.items() if k != field}):
        raise ValueError("Integrity differs")


@pytest.fixture
def setup(tmp_path):
    output = tmp_path / "output"
    states = ["base_14b"] + [f"rot13_14b_level_{level:02d}" for level in range(1, 11)]
    protocols = [{"protocol_hash": content_hash(state), "state_id": state} for state in states]
    plan = seal({"order": [{"state_id": state, "protocol_hash": p["protocol_hash"],
                           "level": index or None} for index, (state, p) in enumerate(zip(states, protocols))],
                 "source_code_commit": "a" * 40, "panel_path": str(tmp_path / "panel.json"),
                 "adapters_manifest_path": str(tmp_path / "adapters.json"),
                 "model_cache": str(tmp_path / "models"), "output_root": str(output),
                 "generation_profile": "long_32000", "save_to_drive": True}, "plan_hash")
    campaign = output / "campaigns" / plan["plan_hash"]
    atomic_write_json(output / "campaigns/plan.json", plan)
    atomic_write_json(campaign / "plan.json", plan)
    args = SimpleNamespace(source_root=tmp_path, plan=campaign / "plan.json", output_root=output,
                           after_state=states[6], priority_state=states[10], execution_order=None)
    calls, barriers = [], []

    def barrier(protocol, root, **kwargs):
        barriers.append(protocol["state_id"])
        assert kwargs == {"save_to_drive": True, "source_code_commit": "a" * 40}
        return read_json(Path(root) / protocol["protocol_hash"] / "completion_receipt.json")

    def freeze_command(args, job, protocol_path, baseline_path):
        return [sys.executable, str(tmp_path / "scripts/run_supplemental_gpqa.py"), "freeze",
                "--output", str(protocol_path), "--baseline-protocol", str(baseline_path),
                "--state", job["state_id"]]

    frozen = SimpleNamespace(seal=seal, sealed=sealed, content_hash=content_hash, file_hash=file_hash,
        read_json=read_json, atomic_write_json=atomic_write_json, file_lock=file_lock,
        completion_barrier=barrier, freeze_command=freeze_command,
        run_command=lambda args, path: [sys.executable, str(tmp_path / "scripts/run_supplemental_gpqa.py"),
                                       "run", "--protocol", str(path), "--save-to-drive"],
        campaign_plan=lambda args: (plan, protocols))
    for state, protocol in zip(states[:7], protocols[:7]):
        atomic_write_json(campaign / "protocols" / (state + ".json"), protocol)
        atomic_write_json(output / protocol["protocol_hash"] / "completion_receipt.json",
                          {"receipt_hash": content_hash(state + "receipt"), "drive_status": "VERIFIED"})
    return SimpleNamespace(args=args, plan=plan, protocols=protocols, frozen=frozen,
                           campaign=campaign, calls=calls, barriers=barriers)


class HandedOff(BaseException):
    pass


def children(monkeypatch, setup, *, failure=0, signal_during_run=False):
    by_state = {p["state_id"]: p for p in setup.protocols}

    class Child:
        def __init__(self, command, **kwargs):
            self.command, self.signals = command, []
            setup.calls.append(command)
            assert kwargs["cwd"] == setup.args.source_root
            if command[2] == "freeze":
                path = command[command.index("--output") + 1]
                atomic_write_json(path, by_state[command[command.index("--state") + 1]])

        def poll(self):
            return None

        def send_signal(self, number):
            self.signals.append(number)

        def wait(self):
            if self.command[2] == "run":
                if signal_during_run:
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    assert self.signals == [signal.SIGTERM]
                    return -signal.SIGTERM
                if failure:
                    return failure
                protocol = read_json(self.command[self.command.index("--protocol") + 1])
                atomic_write_json(setup.args.output_root / protocol["protocol_hash"] / "completion_receipt.json",
                    {"receipt_hash": content_hash("priority receipt"), "drive_status": "VERIFIED"})
            return 0

    monkeypatch.setattr(PRIORITY.subprocess, "Popen", Child)
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    handed_off = []

    def execute(executable, command):
        handed_off.append((executable, command))
        raise HandedOff

    monkeypatch.setattr(PRIORITY.os, "execv", execute)
    return handed_off


def test_order_amendment_is_separate_exact_permutation_and_original_plan_unchanged(setup):
    before = deepcopy(setup.plan)
    order = PRIORITY.execution_order(setup.frozen, setup.plan, setup.args.after_state, setup.args.priority_state)
    assert order["state_ids"] == ["base_14b"] + [f"rot13_14b_level_{n:02d}" for n in (1, 2, 3, 4, 5, 6, 10, 7, 8, 9)]
    assert set(order["state_ids"]) == {j["state_id"] for j in setup.plan["order"]}
    sealed(order, "order_hash")
    assert order["scientific_protocols_changed"] is False
    assert setup.plan == before


@pytest.mark.parametrize("after,priority", [("missing", "rot13_14b_level_10"),
    ("rot13_14b_level_06", "missing"), ("rot13_14b_level_06", "base_14b"),
    ("rot13_14b_level_06", "rot13_14b_level_06"),
    ("rot13_14b_level_06", "rot13_14b_level_02")])
def test_invalid_priority_never_starts_child(setup, after, priority):
    with pytest.raises(ValueError):
        PRIORITY.execution_order(setup.frozen, setup.plan, after, priority)


def test_priority_uses_frozen_commands_then_execs_original_and_restart_reuses_receipt(setup, monkeypatch):
    handoff = children(monkeypatch, setup)
    immutable_before = {p: p.read_bytes() for p in (setup.args.plan, setup.args.output_root / "campaigns/plan.json")}
    with pytest.raises(HandedOff):
        PRIORITY.run_priority(setup.args, setup.frozen, setup.plan, setup.protocols)
    assert [c[2] for c in setup.calls] == ["freeze", "run"]
    assert setup.barriers == [j["state_id"] for j in setup.plan["order"][:7]] + [setup.args.priority_state]
    expected = [sys.executable, str(setup.args.source_root / "scripts/run_gpqa_sft_campaign.py"),
        "--panel", setup.plan["panel_path"], "--adapters-manifest", setup.plan["adapters_manifest_path"],
        "--model-cache", setup.plan["model_cache"], "--output-root", setup.plan["output_root"],
        "--generation-profile", "long_32000", "--save-to-drive"]
    assert handoff == [(sys.executable, expected)]
    assert not (setup.campaign / "completion_receipt.json").exists()
    for path, before in immutable_before.items():
        assert path.read_bytes() == before
    order_path = next(setup.campaign.glob("execution_order_*.json"))
    receipt_path = next(setup.campaign.glob("priority_execution_*.json"))
    order_bytes, receipt_bytes = order_path.read_bytes(), receipt_path.read_bytes()
    with pytest.raises(HandedOff):
        PRIORITY.run_priority(setup.args, setup.frozen, setup.plan, setup.protocols)
    assert len(setup.calls) == 2  # Neither freeze nor inference is repeated.
    assert len(handoff) == 2
    assert order_path.read_bytes() == order_bytes and receipt_path.read_bytes() == receipt_bytes


def test_missing_predecessor_receipt_refuses_priority_generation(setup, monkeypatch):
    children(monkeypatch, setup)
    (setup.args.output_root / setup.protocols[6]["protocol_hash"] / "completion_receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        PRIORITY.run_priority(setup.args, setup.frozen, setup.plan, setup.protocols)
    assert not setup.calls


def test_first_amendment_cannot_claim_a_completed_intervening_state_ran_later(setup, monkeypatch):
    children(monkeypatch, setup)
    atomic_write_json(setup.args.output_root / setup.protocols[7]["protocol_hash"] / "completion_receipt.json",
                      {"receipt_hash": "already completed"})
    with pytest.raises(ValueError, match="already completed before this amendment"):
        PRIORITY.run_priority(setup.args, setup.frozen, setup.plan, setup.protocols)
    assert not setup.calls
    assert not list(setup.campaign.glob("execution_order_*.json"))


def test_supplied_order_cannot_omit_or_duplicate_states(setup, monkeypatch):
    children(monkeypatch, setup)
    order = PRIORITY.execution_order(setup.frozen, setup.plan, setup.args.after_state, setup.args.priority_state)
    order["state_ids"][-1] = order["state_ids"][0]
    order.pop("order_hash")
    setup.args.execution_order = setup.args.source_root / "bad_order.json"
    atomic_write_json(setup.args.execution_order, seal(order, "order_hash"))
    with pytest.raises(ValueError, match="exact sealed permutation"):
        PRIORITY.run_priority(setup.args, setup.frozen, setup.plan, setup.protocols)
    assert not setup.calls


def test_failed_priority_preserves_exit_code_and_does_not_resume_original(setup, monkeypatch):
    handoff = children(monkeypatch, setup, failure=7)
    assert PRIORITY.run_priority(setup.args, setup.frozen, setup.plan, setup.protocols) == 7
    status = read_json(setup.campaign / "status.json")
    assert status["status"] == "FAILED" and status["child_exit_code"] == 7
    assert not handoff
    assert not list(setup.campaign.glob("priority_execution_*.json"))


def test_signal_forwarded_and_prevents_handoff(setup, monkeypatch):
    handoff = children(monkeypatch, setup, signal_during_run=True)
    assert PRIORITY.run_priority(setup.args, setup.frozen, setup.plan, setup.protocols) == 143
    status = read_json(setup.campaign / "status.json")
    assert status["status"] == "STOPPED"
    assert not handoff


def test_changed_frozen_entrypoint_is_rejected_before_import(tmp_path, monkeypatch):
    for relative in PRIORITY.IMPLEMENTATIONS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise AssertionError('must not import changed source')\n")
    monkeypatch.setenv("SOURCE_CODE_COMMIT", "a" * 40)
    with pytest.raises(ValueError, match="entrypoints differ"):
        PRIORITY.load_frozen(tmp_path, {"source_code_commit": "a" * 40, "implementation_sha256": {}})


def test_main_reconstructs_exact_original_input_bindings_before_any_child(setup, monkeypatch):
    monkeypatch.setattr(PRIORITY, "load_frozen", lambda *args: setup.frozen)
    changed = deepcopy(setup.plan)
    changed["panel_sha256"] = "different bytes"
    setup.frozen.campaign_plan = lambda args: (changed, setup.protocols)
    monkeypatch.setattr(PRIORITY, "run_priority", lambda *args: pytest.fail("No child may run"))
    with pytest.raises(ValueError, match="input bindings differ"):
        PRIORITY.main(["--source-root", str(setup.args.source_root), "--plan", str(setup.args.plan),
                       "--output-root", str(setup.args.output_root), "--after-state", setup.args.after_state,
                       "--priority-state", setup.args.priority_state])
