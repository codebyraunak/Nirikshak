import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("=" * 80)
print("HEXA-FORGE: BASELINE ZERO-SHOT BENCHMARK (UNTRAINED QWEN3-1.7B)")
print("=" * 80)

# Ground-truth test suite: exactly 15 test cases
TEST_CASES = [
    {
        "id": 1,
        "type": "Known Cisco",
        "vendor": "cisco",
        "raw_config": "ip ssh version 2",
        "expected": {
            "security_control": "SSH_VERSION",
            "security_category": "session_security",
            "value": 2,
            "security_meaning": "SSH protocol version 2 is enabled",
            "evidence": "ip ssh version 2"
        }
    },
    {
        "id": 2,
        "type": "Known Cisco",
        "vendor": "cisco",
        "raw_config": "ip ssh version 1",
        "expected": {
            "security_control": "SSH_VERSION",
            "security_category": "session_security",
            "value": 1,
            "security_meaning": "SSH protocol version 1 is enabled",
            "evidence": "ip ssh version 1"
        }
    },
    {
        "id": 3,
        "type": "Known Cisco",
        "vendor": "cisco",
        "raw_config": "transport input telnet",
        "expected": {
            "security_control": "TELNET_ENABLED",
            "security_category": "session_security",
            "value": True,
            "security_meaning": "Insecure Telnet protocol is enabled for management access",
            "evidence": "transport input telnet"
        }
    },
    {
        "id": 4,
        "type": "Different Cisco (Negation)",
        "vendor": "cisco",
        "raw_config": "transport input none",
        "expected": {
            "security_control": "TELNET_ENABLED",
            "security_category": "session_security",
            "value": False,
            "security_meaning": "Telnet management access is disabled",
            "evidence": "transport input none"
        }
    },
    {
        "id": 5,
        "type": "Known Cisco",
        "vendor": "cisco",
        "raw_config": "ip http server",
        "expected": {
            "security_control": "HTTP_MANAGEMENT",
            "security_category": "management_plane",
            "value": True,
            "security_meaning": "Unencrypted HTTP web management server is enabled",
            "evidence": "ip http server"
        }
    },
    {
        "id": 6,
        "type": "Different Cisco (Negation)",
        "vendor": "cisco",
        "raw_config": "no ip http server",
        "expected": {
            "security_control": "HTTP_MANAGEMENT",
            "security_category": "management_plane",
            "value": False,
            "security_meaning": "Unencrypted HTTP web management server is disabled",
            "evidence": "no ip http server"
        }
    },
    {
        "id": 7,
        "type": "Known Cisco",
        "vendor": "cisco",
        "raw_config": "exec-timeout 10 0",
        "expected": {
            "security_control": "SESSION_TIMEOUT",
            "security_category": "session_security",
            "value": 10,
            "security_meaning": "CLI session idle timeout is set to 10 minutes",
            "evidence": "exec-timeout 10 0"
        }
    },
    {
        "id": 8,
        "type": "Different Cisco (AAA)",
        "vendor": "cisco",
        "raw_config": "aaa new-model",
        "expected": {
            "security_control": "AAA_ENABLED",
            "security_category": "access_control",
            "value": True,
            "security_meaning": "AAA authentication framework is globally enabled",
            "evidence": "aaa new-model"
        }
    },
    {
        "id": 9,
        "type": "Juniper",
        "vendor": "juniper",
        "raw_config": "set system services ssh protocol-version v2",
        "expected": {
            "security_control": "SSH_VERSION",
            "security_category": "session_security",
            "value": 2,
            "security_meaning": "SSH protocol version 2 is configured",
            "evidence": "set system services ssh protocol-version v2"
        }
    },
    {
        "id": 10,
        "type": "Juniper",
        "vendor": "juniper",
        "raw_config": "set system services web-management http disable",
        "expected": {
            "security_control": "HTTP_MANAGEMENT",
            "security_category": "management_plane",
            "value": False,
            "security_meaning": "HTTP web management service is disabled",
            "evidence": "set system services web-management http disable"
        }
    },
    {
        "id": 11,
        "type": "Juniper",
        "vendor": "juniper",
        "raw_config": "set system login idle-timeout 15",
        "expected": {
            "security_control": "SESSION_TIMEOUT",
            "security_category": "session_security",
            "value": 15,
            "security_meaning": "User login session idle timeout is configured to 15 minutes",
            "evidence": "set system login idle-timeout 15"
        }
    },
    {
        "id": 12,
        "type": "Arista",
        "vendor": "arista",
        "raw_config": "management ssh protocol-version 2",
        "expected": {
            "security_control": "SSH_VERSION",
            "security_category": "session_security",
            "value": 2,
            "security_meaning": "Management SSH protocol version is restricted to version 2",
            "evidence": "management ssh protocol-version 2"
        }
    },
    {
        "id": 13,
        "type": "Fortinet",
        "vendor": "fortinet",
        "raw_config": "set admin-telnet disable",
        "expected": {
            "security_control": "TELNET_ENABLED",
            "security_category": "session_security",
            "value": False,
            "security_meaning": "Administrative Telnet management access is disabled",
            "evidence": "set admin-telnet disable"
        }
    },
    {
        "id": 14,
        "type": "Fortinet",
        "vendor": "fortinet",
        "raw_config": "set admintimeout 10",
        "expected": {
            "security_control": "SESSION_TIMEOUT",
            "security_category": "session_security",
            "value": 10,
            "security_meaning": "Administrator console idle timeout is set to 10 minutes",
            "evidence": "set admintimeout 10"
        }
    },
    {
        "id": 15,
        "type": "Ambiguous / Non-Security",
        "vendor": "generic",
        "raw_config": "interface GigabitEthernet0/1 description UPLINK_TO_CORE_SWITCH",
        "expected": {
            "security_control": "UNKNOWN",
            "security_category": "none",
            "value": None,
            "security_meaning": "Interface description string; no security control affected",
            "evidence": "interface GigabitEthernet0/1 description UPLINK_TO_CORE_SWITCH"
        }
    }
]

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

model_id = "Qwen/Qwen3-1.7B"

print("1. Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("2. Configuring BitsAndBytes NF4 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("3. Loading base Qwen3-1.7B model onto GPU...")
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

print("\n4. Running Benchmark across 15 cases (with thinking disabled via template)...")

results = []
latencies = []

for case in TEST_CASES:
    raw_cfg = case["raw_config"]
    user_content = f"Configuration: {raw_cfg}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    # Use enable_thinking=False to suppress thinking output
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
    
    # Strip any potential residual thinking or markdown code blocks
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
        
    # Evaluate metrics
    exp = case["expected"]
    
    pred_control = str(parsed_json.get("security_control", "")).strip().upper()
    exp_control = str(exp["security_control"]).strip().upper()
    control_match = (pred_control == exp_control)
    
    # Value normalization comparison
    pred_val = parsed_json.get("value")
    exp_val = exp["value"]
    val_match = False
    if pred_val is not None and exp_val is not None:
        # Check boolean or integer equivalence
        if isinstance(exp_val, bool):
            val_match = (str(pred_val).strip().lower() == str(exp_val).lower())
        elif isinstance(exp_val, (int, float)):
            try:
                val_match = (int(float(pred_val)) == int(exp_val))
            except Exception:
                val_match = False
        else:
            val_match = (str(pred_val).strip().lower() == str(exp_val).strip().lower())
    elif pred_val is None and exp_val is None:
        val_match = True
        
    pred_cat = str(parsed_json.get("security_category", "")).strip().lower()
    exp_cat = str(exp["security_category"]).strip().lower()
    cat_match = (pred_cat == exp_cat)
    
    pred_evidence = str(parsed_json.get("evidence", "")).strip()
    exp_evidence = str(exp["evidence"]).strip()
    evidence_match = (pred_evidence == exp_evidence)
    
    meaning_valid = bool(parsed_json.get("security_meaning", "").strip())
    
    full_pass = json_valid and control_match and val_match and evidence_match
    
    res_entry = {
        "id": case["id"],
        "type": case["type"],
        "raw_config": raw_cfg,
        "expected_control": exp_control,
        "predicted_control": pred_control,
        "expected_value": exp_val,
        "predicted_value": pred_val,
        "json_valid": json_valid,
        "control_match": control_match,
        "value_match": val_match,
        "category_match": cat_match,
        "evidence_match": evidence_match,
        "meaning_valid": meaning_valid,
        "pass": full_pass,
        "latency_sec": round(lat, 3),
        "raw_model_output": raw_output,
        "parsed_json": parsed_json
    }
    results.append(res_entry)

peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
avg_latency = sum(latencies) / len(latencies)

# Aggregate Metrics
total = len(results)
valid_json_count = sum(1 for r in results if r["json_valid"])
control_count = sum(1 for r in results if r["control_match"])
value_count = sum(1 for r in results if r["value_match"])
cat_count = sum(1 for r in results if r["category_match"])
evidence_count = sum(1 for r in results if r["evidence_match"])
meaning_count = sum(1 for r in results if r["meaning_valid"])
pass_count = sum(1 for r in results if r["pass"])

summary = {
    "model": model_id,
    "quantization": "4-bit NF4 double quant",
    "total_cases": total,
    "valid_json_percentage": round((valid_json_count / total) * 100, 1),
    "control_accuracy_percentage": round((control_count / total) * 100, 1),
    "value_accuracy_percentage": round((value_count / total) * 100, 1),
    "category_accuracy_percentage": round((cat_count / total) * 100, 1),
    "evidence_accuracy_percentage": round((evidence_count / total) * 100, 1),
    "meaning_valid_percentage": round((meaning_count / total) * 100, 1),
    "overall_exact_match_percentage": round((pass_count / total) * 100, 1),
    "average_inference_latency_sec": round(avg_latency, 3),
    "model_load_vram_mb": round(load_vram_mb, 1),
    "peak_vram_mb": round(peak_vram_mb, 1)
}

output_payload = {
    "summary": summary,
    "details": results
}

os.makedirs("outputs", exist_ok=True)
with open("outputs/baseline_results.json", "w") as f:
    json.dump(output_payload, f, indent=2)

print("\n" + "=" * 90)
print(f"{'ID':<3} | {'CONFIG':<32} | {'EXP CONTROL':<15} | {'PRED CONTROL':<15} | {'VAL MATCH':<9} | {'JSON':<6} | {'STATUS'}")
print("-" * 90)
for r in results:
    cfg_display = (r["raw_config"][:29] + "...") if len(r["raw_config"]) > 32 else r["raw_config"]
    status = "PASS" if r["pass"] else "FAIL"
    val_status = "OK" if r["value_match"] else "ERR"
    json_status = "OK" if r["json_valid"] else "FAIL"
    print(f"{r['id']:<3} | {cfg_display:<32} | {r['expected_control']:<15} | {r['predicted_control'][:15]:<15} | {val_status:<9} | {json_status:<6} | {status}")
print("=" * 90)

print("\nAGGREGATE METRICS:")
for k, v in summary.items():
    print(f"  {k}: {v}")
print("\nResults saved to outputs/baseline_results.json")
