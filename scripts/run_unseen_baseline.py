import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("=" * 80)
print("HEXA-FORGE PHASE 1: UNSEEN TEST SET BASELINE EVALUATION (BASE QWEN3-1.7B)")
print("=" * 80)

SYSTEM_PROMPT = """You are a network security semantic parser.
Do not decide compliance.
Interpret the configuration and map it to the closest security control.
Return ONLY valid JSON.
Never copy the raw configuration into security_control.
The evidence field must reproduce the supplied configuration exactly.

Supported canonical security_controls:
- SSH_VERSION (value: 1, 2)
- TELNET_ENABLED (value: true, false)
- HTTP_MANAGEMENT (value: true, false)
- SESSION_TIMEOUT (value: integer minutes)
- AAA_ENABLED (value: true, false)
- UNKNOWN (value: null, if configuration does not relate to network security)

Output Schema:
{
  "security_control": "SSH_VERSION | TELNET_ENABLED | HTTP_MANAGEMENT | SESSION_TIMEOUT | AAA_ENABLED | UNKNOWN",
  "security_category": "session_security | management_plane | access_control | none",
  "value": 2 | true | false | 10 | null,
  "security_meaning": "concise description of the security state created",
  "evidence": "EXACT raw configuration string provided by user"
}
Output strictly valid JSON and nothing else."""

# 1. Load test cases from test_unseen.jsonl ONLY (NO ground truth access during inference)
test_input_file = "data/test_unseen.jsonl"
print(f"Loading input cases strictly from: {test_input_file}")
unseen_cases = []
with open(test_input_file, "r", encoding="utf-8") as f:
    for line in f:
        unseen_cases.append(json.loads(line))

print(f"Loaded {len(unseen_cases)} test cases.")

# 2. Setup Quantized Model
model_id = "Qwen/Qwen3-1.7B"
print(f"\nLoading Tokenizer: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Configuring BitsAndBytes NF4 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading base Qwen3-1.7B model onto GPU...")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
load_time = time.time() - t0
load_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
print(f"Model loaded in {load_time:.2f}s. Allocated VRAM: {load_vram_mb:.1f} MiB")

# 3. Run Inference on test_unseen.jsonl
print("\nRunning Inference across 20 unseen cases (thinking suppressed)...")
predictions = []
latencies = []

for case in unseen_cases:
    case_id = case["id"]
    raw_cfg = case["raw_config"]
    user_content = f"Configuration: {raw_cfg}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    
    t_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    lat = time.time() - t_start
    latencies.append(lat)
    
    gen_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    cleaned_output = raw_output
    if "</think>" in cleaned_output:
        cleaned_output = cleaned_output.split("</think>")[-1].strip()
    if "```json" in cleaned_output:
        cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned_output:
        cleaned_output = cleaned_output.split("```")[1].split("```")[0].strip()
        
    json_valid = False
    parsed_json = {}
    try:
        parsed_json = json.loads(cleaned_output)
        json_valid = True
    except Exception as e:
        json_valid = False
        
    pred_record = {
        "id": case_id,
        "vendor": case.get("vendor"),
        "platform": case.get("platform"),
        "raw_config": raw_cfg,
        "json_valid": json_valid,
        "security_control": parsed_json.get("security_control") if json_valid else None,
        "security_category": parsed_json.get("security_category") if json_valid else None,
        "value": parsed_json.get("value") if json_valid else None,
        "security_meaning": parsed_json.get("security_meaning") if json_valid else None,
        "evidence": parsed_json.get("evidence") if json_valid else None,
        "latency_sec": round(lat, 3),
        "raw_output": raw_output
    }
    predictions.append(pred_record)
    print(f"  [Case {case_id:2d}/20] Latency: {lat:.2f}s | JSON: {'OK' if json_valid else 'FAIL'} | Control: {pred_record['security_control']}")

peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
avg_latency = sum(latencies) / len(latencies)

# Clean up model from GPU memory before evaluation scoring
del model
torch.cuda.empty_cache()

# Save predictions to outputs/unseen_baseline_results.json
os.makedirs("outputs", exist_ok=True)
output_pred_file = "outputs/unseen_baseline_results.json"
with open(output_pred_file, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "model": model_id,
            "stage": "pre_training_baseline",
            "quantization": "4-bit NF4 double quant",
            "load_vram_mb": round(load_vram_mb, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "average_latency_sec": round(avg_latency, 3)
        },
        "predictions": predictions
    }, f, indent=2)

print(f"\nRaw predictions saved to: {output_pred_file}")

# 4. Score predictions against data/test_unseen_ground_truth.jsonl
print("\n" + "=" * 80)
print("SCORING AGAINST GROUND TRUTH (data/test_unseen_ground_truth.jsonl)...")
print("=" * 80)

gt_file = "data/test_unseen_ground_truth.jsonl"
gt_map = {}
with open(gt_file, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        gt_map[r["id"]] = r

scored_details = []
json_valid_count = 0
control_match_count = 0
category_match_count = 0
value_match_count = 0
evidence_match_count = 0
full_pass_count = 0
unknown_total = 0
unknown_correct = 0

for p in predictions:
    cid = p["id"]
    gt = gt_map[cid]
    
    j_valid = p["json_valid"]
    if j_valid:
        json_valid_count += 1
        
    # Control match
    pred_ctrl = str(p["security_control"]).strip().upper() if p["security_control"] else ""
    exp_ctrl = str(gt["security_control"]).strip().upper()
    ctrl_match = (pred_ctrl == exp_ctrl)
    if ctrl_match:
        control_match_count += 1
        
    # Category match
    pred_cat = str(p["security_category"]).strip().lower() if p["security_category"] else ""
    exp_cat = str(gt["security_category"]).strip().lower()
    cat_match = (pred_cat == exp_cat)
    if cat_match:
        category_match_count += 1
        
    # Value match
    exp_val = gt["value"]
    pred_val = p["value"]
    val_match = False
    if exp_val is None:
        val_match = (pred_val is None)
    elif isinstance(exp_val, bool):
        if isinstance(pred_val, bool):
            val_match = (pred_val == exp_val)
        elif isinstance(pred_val, str):
            val_match = (pred_val.strip().lower() == str(exp_val).lower())
    elif isinstance(exp_val, (int, float)):
        try:
            val_match = (int(float(pred_val)) == int(exp_val))
        except (ValueError, TypeError):
            val_match = False
    else:
        val_match = (str(pred_val).strip().lower() == str(exp_val).strip().lower())
        
    if val_match:
        value_match_count += 1
        
    # Evidence match
    pred_ev = str(p["evidence"]).strip() if p["evidence"] else ""
    exp_ev = str(gt["evidence"]).strip()
    ev_match = (pred_ev == exp_ev)
    if ev_match:
        evidence_match_count += 1
        
    # UNKNOWN accuracy
    if exp_ctrl == "UNKNOWN":
        unknown_total += 1
        if ctrl_match and val_match:
            unknown_correct += 1
            
    # Full pass
    passed = j_valid and ctrl_match and cat_match and val_match and ev_match
    if passed:
        full_pass_count += 1
        
    scored_details.append({
        "id": cid,
        "vendor": p["vendor"],
        "platform": p["platform"],
        "raw_config": p["raw_config"],
        "expected_control": exp_ctrl,
        "predicted_control": pred_ctrl,
        "expected_value": exp_val,
        "predicted_value": pred_val,
        "expected_category": exp_cat,
        "predicted_category": pred_cat,
        "json_valid": j_valid,
        "control_match": ctrl_match,
        "category_match": cat_match,
        "value_match": val_match,
        "evidence_match": ev_match,
        "pass": passed,
        "latency_sec": p["latency_sec"]
    })

total_cases = len(predictions)
metrics = {
    "total_cases": total_cases,
    "valid_json_count": json_valid_count,
    "valid_json_percentage": round((json_valid_count / total_cases) * 100, 1),
    "security_control_accuracy_percentage": round((control_match_count / total_cases) * 100, 1),
    "security_category_accuracy_percentage": round((category_match_count / total_cases) * 100, 1),
    "value_accuracy_percentage": round((value_match_count / total_cases) * 100, 1),
    "evidence_exact_match_percentage": round((evidence_match_count / total_cases) * 100, 1),
    "full_semantic_pass_rate_percentage": round((full_pass_count / total_cases) * 100, 1),
    "unknown_accuracy_percentage": round((unknown_correct / unknown_total * 100) if unknown_total > 0 else 0.0, 1),
    "average_latency_sec": round(avg_latency, 3),
    "peak_vram_mb": round(peak_vram_mb, 1)
}

# Update outputs/unseen_baseline_results.json with scored results
with open(output_pred_file, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "model": model_id,
            "stage": "pre_training_baseline",
            "quantization": "4-bit NF4 double quant",
            "load_vram_mb": round(load_vram_mb, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "average_latency_sec": round(avg_latency, 3)
        },
        "metrics": metrics,
        "details": scored_details
    }, f, indent=2)

print("\n" + "=" * 105)
print(f"{'ID':<3} | {'VENDOR/PLATFORM':<18} | {'EXPECTED':<17} | {'PREDICTED':<17} | {'VAL':<5} | {'JSON':<5} | {'PASS':<5} | {'RAW CONFIG'}")
print("-" * 105)
for d in scored_details:
    vp = f"{d['vendor']} {d['platform']}"[:18]
    cfg_display = d['raw_config'].replace('\n', ' ')[:30]
    val_status = "OK" if d["value_match"] else "ERR"
    j_status = "OK" if d["json_valid"] else "FAIL"
    pass_status = "PASS" if d["pass"] else "FAIL"
    print(f"{d['id']:<3} | {vp:<18} | {d['expected_control']:<17} | {d['predicted_control'][:17]:<17} | {val_status:<5} | {j_status:<5} | {pass_status:<5} | {cfg_display}")
print("=" * 105)

print("\nPHASE 1 SUMMARY METRICS:")
for k, v in metrics.items():
    print(f"  {k:<40}: {v}")
