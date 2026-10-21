import streamlit as st
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from evaluation.run_eval import run_evaluation_pipeline, RESULTS_DIR
from evaluation.fixtures import TEST_FIXTURES
from backend.services.medgemma import medgemma_service

st.set_page_config(
    page_title="Evaluation & Benchmarks | CareFlow",
    page_icon="🧪",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .eval-header {
        background: linear-gradient(135deg, #3B0764 0%, #6B21A8 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="eval-header">
    <h2 style="margin:0; font-size: 24px;">🧪 CareFlow Clinical AI Benchmark Suite</h2>
    <div style="font-size: 14px; opacity: 0.9; margin-top: 4px;">
        Empirical Evaluation of MedGemma Extraction, Deterministic Triage Protocols & Failure Safety
    </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div style="background-color: #FEF3C7; border-left: 4px solid #D97706; padding: 10px 14px; border-radius: 6px; color: #92400E; font-size: 13px; margin-bottom: 20px;">
    <strong>Benchmarking Disclaimer:</strong> This trial measures unfine-tuned Google MedGemma performance on permitted test fixtures.
    This evaluation is for engineering and research validation only and <strong>does NOT constitute clinical validation</strong>.
</div>
""", unsafe_allow_html=True)

st.write(f"**Current Model Service Mode:** `{medgemma_service.mode.upper()}` | **Model ID:** `{medgemma_service.model_id}`")

from evaluation.eval_policies import run_policy_benchmark

tab1, tab2 = st.tabs(["📋 Clinical Pipeline Fixtures (7 Tests)", "📊 Multi-Policy Questioning Benchmark (Budgets 4/6/8)"])

# =====================================================================
# TAB 1: CLINICAL PIPELINE FIXTURES
# =====================================================================
with tab1:
    col_run, col_info = st.columns([1, 2])
    with col_run:
        if st.button("▶️ Run Pipeline Fixture Suite", type="primary", key="btn_run_pipeline", use_container_width=True):
            with st.spinner("Executing benchmark across test fixtures..."):
                summary_res = run_evaluation_pipeline()
                st.session_state.last_eval = summary_res
                st.success("Pipeline benchmark completed successfully!")

    with col_info:
        st.caption(
            "Runs all 7 standardized fixtures: Acute Coronary P0, Meningeal P1, Severe Pain P1, "
            "Contradictory Duration P2, Unknown Dimension P3, Routine Cold P3, and Simulated Model Failure ERR."
        )

    st.write("---")

    eval_data = getattr(st.session_state, "last_eval", None)
    if not eval_data:
        pipeline_files = [p for p in RESULTS_DIR.glob("eval_20*_summary.json")]
        if pipeline_files:
            pipeline_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            try:
                with open(pipeline_files[0], "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                    st.caption(f"Displaying latest saved run: `{pipeline_files[0].name}`")
            except Exception:
                pass

    if eval_data:
        st.subheader(f"Pipeline Performance Summary (Run: {eval_data.get('run_id')})")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Triage Accuracy", f"{eval_data.get('triage_accuracy_pct', 0)}%")
        m2.metric("Extraction Accuracy", f"{eval_data.get('extraction_accuracy_pct', 0)}%")
        m3.metric("Conflict Preservation", f"{eval_data.get('conflict_accuracy_pct', 0)}%")
        m4.metric("Unknown Preservation", f"{eval_data.get('unknown_accuracy_pct', 0)}%")
        m5.metric("Schema Error Rate", f"{eval_data.get('schema_error_rate_pct', 0)}%")
        m6.metric("Mean Latency", f"{eval_data.get('mean_latency_ms', 0)} ms")

        st.write("---")
        st.subheader("Detailed Fixture-by-Fixture Results")

        cases = eval_data.get("cases", [])
        if cases:
            table_rows = []
            for c in cases:
                match_icon = "✅" if c.get("priority_match") else "❌"
                table_rows.append({
                    "Fixture ID": c.get("fixture_id"),
                    "Scenario Title": c.get("title"),
                    "Expected Priority": c.get("expected_priority"),
                    "Assigned Priority": c.get("assigned_priority"),
                    "Match": f"{match_icon} {c.get('priority_match')}",
                    "Rules Triggered": ", ".join(c.get("triggered_rule_ids", [])) or "None",
                    "Schema Valid": "✅ Yes" if c.get("schema_valid") else "❌ No",
                    "Latency (ms)": c.get("latency_ms")
                })

            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True)

            run_id = eval_data.get("run_id")
            jsonl_path = RESULTS_DIR / f"{run_id}.jsonl"
            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    jsonl_bytes = f.read()
                st.download_button(
                    label="📥 Download Pipeline Results (JSONL)",
                    data=jsonl_bytes,
                    file_name=f"{run_id}.jsonl",
                    mime="application/jsonlines"
                )

    with st.expander("📚 Permitted Benchmark Scenarios Specification", expanded=False):
        for f in TEST_FIXTURES:
            st.markdown(f"**{f['id']}**: {f['title']}")
            st.caption(f"{f['description']} (Expected: {f['expected_priority']})")
            st.write(f"- Utterances: `{f['messages']}`")

# =====================================================================
# TAB 2: MULTI-POLICY QUESTIONING BENCHMARK
# =====================================================================
with tab2:
    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        if st.button("▶️ Run Policy Comparison Benchmark", type="primary", key="btn_run_policies", use_container_width=True):
            with st.spinner("Benchmarking 4 Questioning Policies across budgets 4, 6, 8..."):
                pol_summary = run_policy_benchmark()
                st.session_state.last_policy_eval = pol_summary
                st.success("Policy benchmark completed successfully!")

    with p_col2:
        st.caption(
            "Empirically compares Fixed, MedGemma-only, State-aware heuristic, and CareFlow-BQ "
            "across question budgets (4, 6, 8) evaluating extraction quality, critical recall, coverage, and redundancy."
        )

    st.write("---")

    pol_data = getattr(st.session_state, "last_policy_eval", None)
    if not pol_data:
        policy_files = [p for p in RESULTS_DIR.glob("eval_policies_*_summary.json")]
        if policy_files:
            policy_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            try:
                with open(policy_files[0], "r", encoding="utf-8") as f:
                    pol_data = json.load(f)
                    st.caption(f"Displaying latest saved run: `{policy_files[0].name}`")
            except Exception:
                pass

    if pol_data:
        st.subheader(f"Multi-Policy Comparative Performance (Run: {pol_data.get('run_id')})")
        agg = pol_data.get("aggregated_results", {})

        agg_rows = []
        for p_name, b_dict in agg.items():
            for b_key, stats in b_dict.items():
                b_num = b_key.replace("budget_", "")
                agg_rows.append({
                    "Policy": p_name.upper(),
                    "Question Budget": int(b_num),
                    "Avg Questions Asked": stats.get("avg_questions"),
                    "Redundancy Rate": f"{round(stats.get('avg_redundancy_rate', 0)*100, 1)}%",
                    "Critical Question Recall": f"{round(stats.get('avg_critical_recall', 0)*100, 1)}%",
                    "Info Coverage": f"{round(stats.get('avg_info_coverage', 0)*100, 1)}%",
                    "Extraction Quality": f"{round(stats.get('avg_extraction_quality', 0)*100, 1)}%",
                    "Missed Escalations": stats.get("total_missed_escalations", 0),
                    "Inappropriate Stops": stats.get("total_inappropriate_stops", 0),
                    "Avg Turn Latency (ms)": stats.get("avg_latency_ms", 0)
                })

        if agg_rows:
            agg_df = pd.DataFrame(agg_rows)
            st.dataframe(agg_df, use_container_width=True)

        p_run_id = pol_data.get("run_id")
        p_jsonl_path = RESULTS_DIR / f"{p_run_id}.jsonl"
        if p_jsonl_path.exists():
            with open(p_jsonl_path, "r", encoding="utf-8") as f:
                p_jsonl_bytes = f.read()
            st.download_button(
                label="📥 Download Multi-Policy Interaction Traces (JSONL)",
                data=p_jsonl_bytes,
                file_name=f"{p_run_id}.jsonl",
                mime="application/jsonlines"
            )

        with st.expander("🔬 Individual Scenario Breakdown", expanded=False):
            case_sums = pol_data.get("case_summaries", [])
            if case_sums:
                st.dataframe(pd.DataFrame(case_sums), use_container_width=True)
