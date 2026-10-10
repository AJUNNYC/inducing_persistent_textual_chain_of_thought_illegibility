import pytest
from shared_procedures.storage import read_json, atomic_write_json
from training.rl_grpo.training_cycle.rollouts.rollout_collection import collect_rollouts,validate_rollout_groups,format_rl_prompt
from tests.unit.training.rl_grpo.setup.test_rl_setup import fixture_state,fixture_manifest


def test_actual_8_by_4_logged_order_and_cached_no_resampling(tmp_path):
    state=fixture_state(tmp_path)
    questions=fixture_manifest()["questions"][:8]
    first=collect_rollouts(state,questions)
    assert state.runtime.samples==32
    assert [r["question_id"] for r in first] == [f"q{i}" for i in range(8) for _ in range(4)]
    assert {r["completion_index"] for r in first} == {0,1,2,3}
    second=collect_rollouts(state,questions)
    assert first==second and state.runtime.samples==32
    assert all("raw_output_hash" in r for r in first)
    assert "readability" not in format_rl_prompt("a public problem")


def test_invalid_count_and_duplicate_group_rejected(tmp_path):
    state=fixture_state(tmp_path)
    records=collect_rollouts(state,fixture_manifest()["questions"][:8])
    with pytest.raises(ValueError): validate_rollout_groups(records[:-1])
    records[1]["completion_id"]=records[0]["completion_id"]
    with pytest.raises(ValueError): validate_rollout_groups(records)


def test_corrupt_cache_not_resampled(tmp_path):
    state=fixture_state(tmp_path)
    questions=fixture_manifest()["questions"][:8]
    collect_rollouts(state,questions)
    path=next((tmp_path/"rollouts/step_001").glob("*.json"))
    record=read_json(path)
    record["raw_output"]="changed"
    atomic_write_json(path,record)
    with pytest.raises(ValueError):collect_rollouts(state,questions)
    assert state.runtime.samples==32
