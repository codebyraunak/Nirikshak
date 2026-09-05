import json
import sys
from collections import Counter

print("=" * 80)
print("HEXA-FORGE: UNSEEN TEST SET INTEGRITY & LEAKAGE VALIDATION")
print("=" * 80)

# Load test files
with open("data/test_unseen_ground_truth.jsonl", "r", encoding="utf-8") as f:
    gt_records = [json.loads(line) for line in f]

with open("data/test_unseen.jsonl", "r", encoding="utf-8") as f:
    input_records = [json.loads(line) for line in f]

# 1. Exactly 20 records
print(f"[1] Record Count: Ground Truth = {len(gt_records)}, Input = {len(input_records)}")
assert len(gt_records) == 20, "Expected exactly 20 ground truth records!"
assert len(input_records) == 20, "Expected exactly 20 input records!"
print("  PASS: Exactly 20 records in both files.")

# 2. Check fields & types in Ground Truth
print("\n[2] Checking Ground Truth Schema & Data Types...")
REQUIRED_FIELDS = [
    "vendor", "platform", "raw_config", "security_control",
    "security_category", "value", "security_meaning", "evidence",
    "source_type", "source_url"
]
SUPPORTED_CONTROLS = ["SSH_VERSION", "TELNET_ENABLED", "HTTP_MANAGEMENT", "SESSION_TIMEOUT", "AAA_ENABLED", "UNKNOWN"]

for idx, r in enumerate(gt_records, 1):
    for field in REQUIRED_FIELDS:
        assert field in r, f"Record {idx} missing field {field}"
    assert r["security_control"] in SUPPORTED_CONTROLS, f"Record {idx} invalid control: {r['security_control']}"
    assert r["evidence"] == r["raw_config"], f"Record {idx} evidence != raw_config"
    assert r["source_url"].startswith("http"), f"Record {idx} invalid source_url: {r['source_url']}"
    
    ctrl = r["security_control"]
    val = r["value"]
    if ctrl == "UNKNOWN":
        assert val is None, f"Record {idx} UNKNOWN must have value None"
    elif ctrl in ["TELNET_ENABLED", "HTTP_MANAGEMENT", "AAA_ENABLED"]:
        assert isinstance(val, bool), f"Record {idx} {ctrl} must have boolean value"
    elif ctrl in ["SSH_VERSION", "SESSION_TIMEOUT"]:
        assert isinstance(val, int), f"Record {idx} {ctrl} must have integer value"
print("  PASS: All schema, control, value type, and evidence assertions verified.")

# 3. Check for duplicates within test set
print("\n[3] Checking for Internal Duplicates...")
test_configs = [r["raw_config"].strip().lower() for r in gt_records]
dupes = [c for c, count in Counter(test_configs).items() if count > 1]
assert len(dupes) == 0, f"Duplicate raw_configs found in test set: {dupes}"
print("  PASS: 0 internal duplicates. All 20 test raw_configs are unique.")

# 4. Check for overlap / leakage with Train, Validation, and Baseline Benchmark
print("\n[4] Checking for Overlap / Data Contamination...")
train_configs = set()
val_configs = set()
with open("data/semantic_mappings.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        cfg = item["raw_config"].strip().lower()
        if item["split"] == "train":
            train_configs.add(cfg)
        else:
            val_configs.add(cfg)

baseline_configs = set()
with open("outputs/baseline_results.json", "r", encoding="utf-8") as f:
    base_data = json.loads(f.read())
    for item in base_data["details"]:
        baseline_configs.add(item["raw_config"].strip().lower())

test_set = set(test_configs)

train_leakage = test_set.intersection(train_configs)
val_leakage = test_set.intersection(val_configs)
baseline_leakage = test_set.intersection(baseline_configs)

print(f"  Overlap with Train set ({len(train_configs)} records):      {len(train_leakage)}")
print(f"  Overlap with Validation set ({len(val_configs)} records): {len(val_leakage)}")
print(f"  Overlap with Baseline Benchmark ({len(baseline_configs)} records): {len(baseline_leakage)}")

assert len(train_leakage) == 0, f"LEAKAGE DETECTED WITH TRAIN: {train_leakage}"
assert len(val_leakage) == 0, f"LEAKAGE DETECTED WITH VALIDATION: {val_leakage}"
assert len(baseline_leakage) == 0, f"LEAKAGE DETECTED WITH BASELINE: {baseline_leakage}"
print("  PASS: ZERO LEAKAGE! Test set is 100% strictly unseen across all previous datasets.")

# 5. Distribution Breakdown
print("\n[5] Test Set Composition:")
print("-" * 50)
ctrl_counts = Counter(r["security_control"] for r in gt_records)
for c, cnt in sorted(ctrl_counts.items()):
    print(f"   - {c:<20}: {cnt}")

print("\n   Vendor Representation:")
v_counts = Counter(r["vendor"] for r in gt_records)
for v, cnt in sorted(v_counts.items()):
    print(f"   - {v:<20}: {cnt}")

print("\n   Platform Representation:")
p_counts = Counter(f"{r['vendor']} {r['platform']}" for r in gt_records)
for p, cnt in sorted(p_counts.items()):
    print(f"   - {p:<25}: {cnt}")

print("\n   Source Type Breakdown:")
s_counts = Counter(r["source_type"] for r in gt_records)
for s, cnt in sorted(s_counts.items()):
    print(f"   - {s:<25}: {cnt}")

print("=" * 80)
print("UNSEEN TEST SET INTEGRITY VERIFIED: 100% COMPLIANT.")
print("=" * 80)
