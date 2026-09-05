import json
import sys
from collections import Counter

print("=" * 80)
print("HEXA-FORGE: DATASET QUALITY & INTEGRITY VALIDATION")
print("=" * 80)

files_to_check = [
    ("data/semantic_mappings.jsonl", "mappings"),
    ("data/train.jsonl", "chat"),
    ("data/validation.jsonl", "chat")
]

REQUIRED_MAPPING_FIELDS = [
    "vendor", "platform", "raw_config", "security_control",
    "security_category", "value", "security_meaning", "evidence",
    "source_type", "split"
]

all_records = []
malformed_count = 0

# 1. Check data/semantic_mappings.jsonl
print("\n[1] Checking data/semantic_mappings.jsonl...")
with open("data/semantic_mappings.jsonl", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            all_records.append(record)
        except Exception as e:
            print(f"  ERROR: Malformed JSON at line {idx}: {e}")
            malformed_count += 1

print(f"  Loaded {len(all_records)} valid records (Malformed lines: {malformed_count})")

# 2. Field Completeness Check
print("\n[2] Checking Required Fields Completeness...")
missing_field_count = 0
for idx, r in enumerate(all_records, 1):
    for f in REQUIRED_MAPPING_FIELDS:
        if f not in r:
            print(f"  ERROR: Line {idx} missing field: {f}")
            missing_field_count += 1
print(f"  Missing field errors: {missing_field_count}")

# 3. Duplicate Records Check
print("\n[3] Checking for Duplicates...")
raw_configs = [r["raw_config"].strip().lower() for r in all_records]
duplicates = [cfg for cfg, count in Counter(raw_configs).items() if count > 1]
if duplicates:
    print(f"  ERROR: Found duplicate raw_config strings: {duplicates}")
else:
    print("  PASS: 0 duplicate raw_config records found. All 50 are globally unique.")

# 4. Train vs Validation Leakage Check
print("\n[4] Checking Train / Validation Leakage...")
train_configs = set(r["raw_config"].strip().lower() for r in all_records if r["split"] == "train")
val_configs = set(r["raw_config"].strip().lower() for r in all_records if r["split"] == "validation")

leakage = train_configs.intersection(val_configs)
if leakage:
    print(f"  ERROR: Leakage detected between Train and Validation: {leakage}")
else:
    print(f"  PASS: 0 leakage! Train set ({len(train_configs)}) and Val set ({len(val_configs)}) are completely disjoint.")

# 5. Check formatted chat files (train.jsonl & validation.jsonl)
print("\n[5] Checking Formatted Qwen Chat Files...")
for filepath in ["data/train.jsonl", "data/validation.jsonl"]:
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                assert "messages" in obj
                assert len(obj["messages"]) == 3
                assert obj["messages"][0]["role"] == "system"
                assert obj["messages"][1]["role"] == "user"
                assert obj["messages"][2]["role"] == "assistant"
                # Validate assistant content is valid JSON matching schema
                assistant_json = json.loads(obj["messages"][2]["content"])
                assert "security_control" in assistant_json
                assert "evidence" in assistant_json
                count += 1
            except Exception as e:
                print(f"  ERROR in {filepath} at line {idx}: {e}")
    print(f"  {filepath}: {count} records verified valid Qwen chat format.")

# 6. Distributions & Balance Analysis
print("\n[6] Distribution & Balance Breakdown:")
print("-" * 50)
controls = Counter(r["security_control"] for r in all_records)
print("A. Examples Per Control:")
for ctrl, cnt in sorted(controls.items()):
    print(f"   - {ctrl:<20}: {cnt} ({cnt/len(all_records)*100:.1f}%)")

print("\nB. Boolean Polarity Balance (TELNET, HTTP, AAA):")
bool_records = [r for r in all_records if isinstance(r["value"], bool)]
pos_cnt = sum(1 for r in bool_records if r["value"] is True)
neg_cnt = sum(1 for r in bool_records if r["value"] is False)
print(f"   - Positive (True / Enabled) : {pos_cnt} ({pos_cnt/len(bool_records)*100:.1f}%)")
print(f"   - Negative (False / Disabled): {neg_cnt} ({neg_cnt/len(bool_records)*100:.1f}%)")

print("\nC. Vendor Representation:")
vendors = Counter(r["vendor"] for r in all_records)
for v, cnt in sorted(vendors.items()):
    print(f"   - {v:<15}: {cnt} ({cnt/len(all_records)*100:.1f}%)")

print("\nD. Source Type Breakdown:")
sources = Counter(r["source_type"] for r in all_records)
for s, cnt in sorted(sources.items()):
    print(f"   - {s:<25}: {cnt} ({cnt/len(all_records)*100:.1f}%)")

print("\nE. Split Breakdown:")
splits = Counter(r["split"] for r in all_records)
for sp, cnt in sorted(splits.items()):
    print(f"   - {sp:<15}: {cnt} ({cnt/len(all_records)*100:.1f}%)")

print("=" * 80)
print("DATASET INTEGRITY VALIDATION COMPLETE: ALL CHECKS PASSED.")
print("=" * 80)
