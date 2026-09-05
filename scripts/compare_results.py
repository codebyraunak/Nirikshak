import os
import sys
import json

print("=" * 80)
print("HEXA-FORGE PHASE 4: BASE vs QLoRA COMPARATIVE ANALYSIS")
print("=" * 80)

base_file = "outputs/unseen_baseline_results.json"
qlora_file = "outputs/unseen_qlora_results.json"

assert os.path.exists(base_file), f"Missing {base_file}"
assert os.path.exists(qlora_file), f"Missing {qlora_file}"

with open(base_file, "r", encoding="utf-8") as f:
    base_data = json.load(f)

with open(qlora_file, "r", encoding="utf-8") as f:
    qlora_data = json.load(f)

bm = base_data["metrics"]
qm = qlora_data["metrics"]

base_meta = base_data.get("metadata", {})
qlora_meta = qlora_data.get("metadata", {})

metrics_table = [
    ("JSON Validity (%)", bm["valid_json_percentage"], qm["valid_json_percentage"]),
    ("Security Control Accuracy (%)", bm["security_control_accuracy_percentage"], qm["security_control_accuracy_percentage"]),
    ("Security Category Accuracy (%)", bm["security_category_accuracy_percentage"], qm["security_category_accuracy_percentage"]),
    ("Value Extraction Accuracy (%)", bm["value_accuracy_percentage"], qm["value_accuracy_percentage"]),
    ("Evidence Exact-Match Accuracy (%)", bm["evidence_exact_match_percentage"], qm["evidence_exact_match_percentage"]),
    ("Full Semantic Pass Rate (%)", bm["full_semantic_pass_rate_percentage"], qm["full_semantic_pass_rate_percentage"]),
    ("UNKNOWN Accuracy (%)", bm["unknown_accuracy_percentage"], qm["unknown_accuracy_percentage"]),
    ("Average Inference Latency (s)", bm["average_latency_sec"], qm["average_latency_sec"]),
    ("Peak VRAM Consumption (MiB)", bm["peak_vram_mb"], qm["peak_vram_mb"])
]

comparison_metrics = []
for label, b_val, q_val in metrics_table:
    diff = round(q_val - b_val, 2)
    diff_str = f"{'+' if diff > 0 else ''}{diff}"
    comparison_metrics.append({
        "metric": label,
        "base": b_val,
        "qlora": q_val,
        "change": diff_str
    })

# Case by Case Analysis
b_details = {d["id"]: d for d in base_data["details"]}
q_details = {d["id"]: d for d in qlora_data["details"]}

base_wrong_qlora_correct = []
base_correct_qlora_wrong = []
both_wrong = []
both_correct = []

for cid in sorted(b_details.keys()):
    b = b_details[cid]
    q = q_details[cid]
    
    b_pass = b["pass"]
    q_pass = q["pass"]
    
    info = {
        "id": cid,
        "vendor": b["vendor"],
        "platform": b["platform"],
        "raw_config": b["raw_config"],
        "expected_control": b["expected_control"],
        "expected_value": b["expected_value"],
        "base_pred_control": b["predicted_control"],
        "base_pred_value": b["predicted_value"],
        "base_pass": b_pass,
        "qlora_pred_control": q["predicted_control"],
        "qlora_pred_value": q["predicted_value"],
        "qlora_pass": q_pass
    }
    
    if not b_pass and q_pass:
        base_wrong_qlora_correct.append(info)
    elif b_pass and not q_pass:
        base_correct_qlora_wrong.append(info)
    elif not b_pass and not q_pass:
        both_wrong.append(info)
    else:
        both_correct.append(info)

comparison_payload = {
    "summary": {
        "total_test_cases": bm["total_cases"],
        "base_full_pass_count": sum(1 for d in base_data["details"] if d["pass"]),
        "qlora_full_pass_count": sum(1 for d in qlora_data["details"] if d["pass"]),
        "improved_cases_count": len(base_wrong_qlora_correct),
        "regressed_cases_count": len(base_correct_qlora_wrong),
        "both_wrong_count": len(both_wrong),
        "both_correct_count": len(both_correct)
    },
    "metrics_comparison": comparison_metrics,
    "case_transitions": {
        "base_wrong_qlora_correct": base_wrong_qlora_correct,
        "base_correct_qlora_wrong": base_correct_qlora_wrong,
        "both_wrong": both_wrong,
        "both_correct": both_correct
    }
}

os.makedirs("outputs", exist_ok=True)
comp_json_file = "outputs/base_vs_qlora_comparison.json"
with open(comp_json_file, "w", encoding="utf-8") as f:
    json.dump(comparison_payload, f, indent=2)

print(f"Comparison JSON saved to: {comp_json_file}")

# Write human-readable report
report_file = "outputs/base_vs_qlora_report.txt"
lines = []
lines.append("=" * 85)
lines.append("HEXA-FORGE: BASE QWEN3-1.7B vs QLoRA ADAPTER COMPARISON REPORT")
lines.append("Evaluation on 20-Case Strictly Held-Out Unseen Test Set")
lines.append("=" * 85)
lines.append("")
lines.append(f"{'Metric':<38} | {'Base':<12} | {'QLoRA':<12} | {'Change':<12}")
lines.append("-" * 85)
for m in comparison_metrics:
    lines.append(f"{m['metric']:<38} | {str(m['base']):<12} | {str(m['qlora']):<12} | {m['change']:<12}")
lines.append("=" * 85)
lines.append("")

lines.append(f"OVERALL SUMMARY:")
lines.append(f"  - Total Unseen Cases Evaluated:       {bm['total_cases']}")
lines.append(f"  - Base Full Semantic Passes:           {comparison_payload['summary']['base_full_pass_count']}/{bm['total_cases']} ({bm['full_semantic_pass_rate_percentage']}%)")
lines.append(f"  - QLoRA Full Semantic Passes:          {comparison_payload['summary']['qlora_full_pass_count']}/{qm['total_cases']} ({qm['full_semantic_pass_rate_percentage']}%)")
diff_pass = round(qm['full_semantic_pass_rate_percentage'] - bm['full_semantic_pass_rate_percentage'], 1)
lines.append(f"  - Net Pass Rate Delta:                 {'+' if diff_pass > 0 else ''}{diff_pass} percentage points")
lines.append(f"  - Improvements (Base Wrong -> QLoRA OK): {len(base_wrong_qlora_correct)}")
lines.append(f"  - Regressions (Base OK -> QLoRA Wrong): {len(base_correct_qlora_wrong)}")
lines.append(f"  - Both Failed:                          {len(both_wrong)}")
lines.append(f"  - Both Passed:                          {len(both_correct)}")
lines.append("")

lines.append("=" * 85)
lines.append("1. BASE WRONG -> QLoRA CORRECT (IMPROVEMENTS):")
lines.append("=" * 85)
if not base_wrong_qlora_correct:
    lines.append("  (None)")
for c in base_wrong_qlora_correct:
    lines.append(f"  [Case {c['id']:2d}] {c['vendor']} {c['platform']} | Raw: {repr(c['raw_config'])}")
    lines.append(f"    Expected: {c['expected_control']} = {c['expected_value']}")
    lines.append(f"    Base:     {c['base_pred_control']} = {c['base_pred_value']} (FAIL)")
    lines.append(f"    QLoRA:    {c['qlora_pred_control']} = {c['qlora_pred_value']} (PASS)")
    lines.append("")

lines.append("=" * 85)
lines.append("2. BASE CORRECT -> QLoRA WRONG (REGRESSIONS):")
lines.append("=" * 85)
if not base_correct_qlora_wrong:
    lines.append("  (None)")
for c in base_correct_qlora_wrong:
    lines.append(f"  [Case {c['id']:2d}] {c['vendor']} {c['platform']} | Raw: {repr(c['raw_config'])}")
    lines.append(f"    Expected: {c['expected_control']} = {c['expected_value']}")
    lines.append(f"    Base:     {c['base_pred_control']} = {c['base_pred_value']} (PASS)")
    lines.append(f"    QLoRA:    {c['qlora_pred_control']} = {c['qlora_pred_value']} (FAIL)")
    lines.append("")

lines.append("=" * 85)
lines.append("3. BOTH WRONG (REMAINING EDGE CASES):")
lines.append("=" * 85)
if not both_wrong:
    lines.append("  (None)")
for c in both_wrong:
    lines.append(f"  [Case {c['id']:2d}] {c['vendor']} {c['platform']} | Raw: {repr(c['raw_config'])}")
    lines.append(f"    Expected: {c['expected_control']} = {c['expected_value']}")
    lines.append(f"    Base:     {c['base_pred_control']} = {c['base_pred_value']}")
    lines.append(f"    QLoRA:    {c['qlora_pred_control']} = {c['qlora_pred_value']}")
    lines.append("")

lines.append("=" * 85)
lines.append("EXPERIMENT INTEGRITY & GENERALIZATION NOTE:")
lines.append("  - Test set data/test_unseen.jsonl was 100% held out during training.")
lines.append("  - No hyperparameter tuning or dataset changes were made against test outcomes.")
lines.append("  - Results are reported strictly for this 20-case test suite without extrapolation.")
lines.append("=" * 85)

report_text = "\n".join(lines)
with open(report_file, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"Human-readable report saved to: {report_file}")
print("\n" + report_text)
