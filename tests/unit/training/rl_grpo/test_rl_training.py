from pathlib import Path
import shutil
import pytest
from shared_procedures.storage import read_json
from training.rl_grpo.setup.rl_setup import MODELS
from training.rl_grpo.rl_training import run_rl_training, verify_all_scientific_snapshots, publish_scientific_snapshot, verify_snapshot
from tests.unit.training.rl_grpo.setup.test_rl_setup import FixtureRuntime, fixture_manifest,fixture_state
from tests.unit.training.rl_grpo.training_cycle.rewards.sonnet_judgment.test_sonnet_opacity_judgment import FixtureJudge


def run_fixture(tmp_path):
    return run_rl_training(fixture_manifest(),tmp_path,judge=FixtureJudge(),
        model_revisions={model:"a"*40 for model in MODELS},runtime_identity={"kind":"fixture"},
        runtime_factory=FixtureRuntime,execution_mode="fixture")


def test_two_independent_trajectories_20_original_snapshots_and_resume(tmp_path):
    FixtureRuntime.created=[]
    result=run_fixture(tmp_path)
    assert result["status"]=="completed" and result["scientific_snapshots"]==20
    assert [r["model_id"] for r in result["runs"]]==list(MODELS)
    engines=FixtureRuntime.created[-2:]
    assert [engine.samples for engine in engines]==[3200,3200]
    assert [engine.update_calls for engine in engines]==[100,100]
    assert engines[0].question_history==engines[1].question_history
    assert all(engine.closed for engine in engines)
    for run in result["runs"]:
        assert [s["step"] for s in run["snapshots"]]==list(range(10,101,10))
        for snapshot in run["snapshots"]:
            assert (Path(snapshot["adapter_path"])/"fixture-adapter.json").read_text()==str(snapshot["step"])
    second=run_fixture(tmp_path)
    assert second["status"]=="completed"
    assert all(engine.samples==0 and engine.update_calls==0 for engine in FixtureRuntime.created[-2:])
    assert read_json(tmp_path/"rl_report.json")["execution_mode"]=="fixture"


def test_missing_snapshot_restored_only_from_original_step(tmp_path):
    result=run_fixture(tmp_path)
    missing=tmp_path/"7b/step_030"
    shutil.rmtree(missing)
    assert run_fixture(tmp_path)["status"]=="completed"
    assert (missing/"fixture-adapter.json").read_text()=="30"
    shutil.rmtree(missing)
    shutil.rmtree(tmp_path/"7b/checkpoints/step_030")
    failure=run_fixture(tmp_path)
    assert failure["status"]=="partial" and failure["failed_model"]==MODELS[0]
    assert not missing.exists()


def test_snapshot_integrity_cannot_pass_corrupt_adapter(tmp_path):
    state=fixture_state(tmp_path)
    state.runtime.step=10
    state.completed_steps=10
    state.save()
    publish_scientific_snapshot(state)
    snapshot=tmp_path/"step_010"
    (snapshot/"fixture-adapter.json").write_text("100")
    with pytest.raises(ValueError):verify_snapshot(snapshot,identity=state.identity,step=10)


def test_mock_judge_cannot_be_labeled_production(tmp_path):
    with pytest.raises(ValueError, match="live, budgeted"):
        run_rl_training(fixture_manifest(), tmp_path, judge=FixtureJudge(),
                        model_revisions={m:"a"*40 for m in MODELS}, runtime_identity={"fixture":True})


def test_selected_14b_stage_runs_only_own_independent_trajectory(tmp_path):
    FixtureRuntime.created=[]
    result=run_rl_training(fixture_manifest(),tmp_path,judge=FixtureJudge(),
        model_revisions={MODELS[1]:"a"*40},runtime_identity={"kind":"fixture"},
        runtime_factory=FixtureRuntime,execution_mode="fixture",model_sizes=("14b",))
    assert result["status"]=="completed" and result["scientific_snapshots"]==10
    assert result["planned_rollouts"]==3200 and result["model_sizes"]==["14b"]
    assert result["complete_trajectory_set"] is False
    assert [r["model_id"] for r in result["runs"]]==[MODELS[1]]
    assert len(FixtureRuntime.created)==1 and not (tmp_path/"7b").exists()


@pytest.mark.parametrize("selection", [(),("7b","7b"),("other",)])
def test_invalid_stage_selector_rejected(tmp_path,selection):
    with pytest.raises(ValueError,match="nonempty unique subset"):
        run_rl_training(fixture_manifest(),tmp_path,judge=FixtureJudge(),
            model_revisions={m:"a"*40 for m in MODELS},runtime_identity={"kind":"fixture"},
            runtime_factory=FixtureRuntime,execution_mode="fixture",model_sizes=selection)
