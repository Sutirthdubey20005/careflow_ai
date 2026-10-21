import pytest
from evaluation.eval_policies import (
    FixedPolicy,
    MedGemmaOnlyPolicy,
    StateAwareHeuristicPolicy,
    CareFlowBQPolicy,
    BenchmarkPatientOracle,
    BENCHMARK_PROFILES,
    run_policy_benchmark
)


def test_benchmark_oracle_slot_resolution_and_redundancy():
    """Test oracle correctly identifies question targets and detects redundancy."""
    profile = BENCHMARK_PROFILES[0]
    oracle = BenchmarkPatientOracle(profile)

    ans1, slot1, is_red1 = oracle.answer_question("When did this chest pain start?")
    assert slot1 == "onset"
    assert is_red1 is False
    assert "45 minutes" in ans1

    # Re-ask onset
    ans2, slot2, is_red2 = oracle.answer_question("How long have you had this?")
    assert slot2 == "onset"
    assert is_red2 is True
    assert "earlier" in ans2.lower()


def test_fixed_policy_budget_adherence():
    """Verify FixedPolicy strictly adheres to question budget."""
    policy = FixedPolicy()
    for budget in [2, 4, 6]:
        questions = []
        for turn in range(budget + 2):
            q = policy.select_question(turn, budget, {}, ["I have pain"])
            if q:
                questions.append(q)
        assert len(questions) == budget


def test_state_aware_policy_terminates_when_slots_full():
    """Verify StateAwareHeuristicPolicy terminates early once all slots are recorded."""
    policy = StateAwareHeuristicPolicy()
    complete_state = {
        "facts": {
            "onset": "1 hour",
            "severity": "5/10",
            "progression": "stable",
            "radiation": "none",
            "associated": "none",
            "negatives": "no fever"
        }
    }
    # Should stop early even if budget is 8
    q = policy.select_question(0, 8, complete_state, ["I have symptoms"])
    assert q is None


def test_careflow_bq_prioritizes_red_flags_for_chest_pain():
    """Verify CareFlow-BQ prioritizes pain radiation/red flag screening for chest pain complaints."""
    policy = CareFlowBQPolicy()
    history = ["I have severe crushing chest pain"]
    empty_state = {"facts": {}}

    first_q = policy.select_question(0, 4, empty_state, history)
    assert first_q is not None
    # Must probe radiation / arm / jaw / shortness of breath
    assert any(term in first_q.lower() for term in ["radiat", "arm", "jaw", "spread", "shortness of breath"])


def test_policy_benchmark_execution():
    """Verify run_policy_benchmark executes end-to-end on budget 4 and generates valid metrics."""
    report = run_policy_benchmark(budgets=[4], output_prefix="test_eval_policies")
    assert "aggregated_results" in report
    assert "fixed" in report["aggregated_results"]
    assert "careflow_bq" in report["aggregated_results"]
    assert len(report["case_summaries"]) > 0

    fixed_b4 = report["aggregated_results"]["fixed"]["budget_4"]
    assert fixed_b4["avg_questions"] == 4.0
    assert 0.0 <= fixed_b4["avg_critical_recall"] <= 1.0
