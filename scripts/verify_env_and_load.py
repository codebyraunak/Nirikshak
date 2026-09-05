import sys
import time
import json

print("=" * 60)
print("HEXA-FORGE: SYSTEM & DEPENDENCY VERIFICATION")
print("=" * 60)

# 1. PyTorch & CUDA
try:
    import torch
    print(f"1. PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"   CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory
        total_mem_mb = total_mem / (1024 ** 2)
        total_mem_gb = total_mem / (1024 ** 3)
        print(f"   Total VRAM: {total_mem_mb:.1f} MiB ({total_mem_gb:.2f} GB)")
    else:
        print("   ERROR: CUDA is not available in PyTorch!")
        sys.exit(1)
except Exception as e:
    print(f"   ERROR checking PyTorch: {e}")
    sys.exit(1)

# 2. bitsandbytes
try:
    import bitsandbytes as bnb
    print(f"2. BitsAndBytes Version: {bnb.__version__}")
except Exception as e:
    print(f"   ERROR importing bitsandbytes: {e}")
    sys.exit(1)

# 3. Transformers
try:
    import transformers
    print(f"3. Transformers Version: {transformers.__version__}")
except Exception as e:
    print(f"   ERROR importing transformers: {e}")
    sys.exit(1)

# 4. PEFT
try:
    import peft
    print(f"4. PEFT Version: {peft.__version__}")
except Exception as e:
    print(f"   ERROR importing peft: {e}")
    sys.exit(1)

# 5. TRL
try:
    import trl
    print(f"5. TRL Version: {trl.__version__}")
except Exception as e:
    print(f"   ERROR importing trl: {e}")
    sys.exit(1)

print("=" * 60)
print("6. MINIMAL Qwen/Qwen3-1.7B 4-BIT MODEL LOADING & INFERENCE TEST")
print("=" * 60)

model_id = "Qwen/Qwen3-1.7B"

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"Downloading/Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print(f"Loading {model_id} with 4-bit NF4 quantization...")
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    load_time = time.time() - start_load
    
    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print("\n>>> MODEL LOADED SUCCESSFULLY! <<<")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Model Load Time: {load_time:.2f} seconds")
    print(f"Allocated VRAM: {allocated_mb:.2f} MiB ({allocated_mb/1024:.2f} GB)")
    print(f"Reserved VRAM:  {reserved_mb:.2f} MiB ({reserved_mb/1024:.2f} GB)")
    print(f"Peak VRAM:      {peak_mb:.2f} MiB ({peak_mb/1024:.2f} GB)")
    print("-" * 60)
    
    # Single test inference
    test_input = "ip ssh version 2"
    system_prompt = (
        "You are a network security semantic interpreter. "
        "Analyze the provided raw network configuration command and output a strictly valid JSON object "
        "with exactly the following fields:\n"
        "{\n"
        '  "security_control": "...",\n'
        '  "security_category": "...",\n'
        '  "value": "...",\n'
        '  "security_meaning": "...",\n'
        '  "evidence": "..."\n'
        "}\n"
        "Do not include any explanation or markdown formatting, output only the JSON object."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Raw configuration: {test_input}"}
    ]
    
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    
    print(f"Running inference for test input: '{test_input}'...")
    start_infer = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    infer_time = time.time() - start_infer
    
    response_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    
    post_infer_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    post_infer_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print(f"\nInference Time: {infer_time:.2f} seconds")
    print(f"VRAM after inference: {post_infer_allocated:.2f} MiB")
    print(f"Peak VRAM during inference: {post_infer_peak:.2f} MiB ({post_infer_peak/1024:.2f} GB)")
    print("-" * 60)
    print("RAW MODEL OUTPUT:")
    print(response_text)
    print("-" * 60)
    
    # Try parsing JSON
    try:
        clean_text = response_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
        parsed = json.loads(clean_text)
        print("JSON PARSING: SUCCESS")
        print(json.dumps(parsed, indent=2))
    except Exception as je:
        print(f"JSON PARSING: FAILED ({je})")
        
    print("=" * 60)
    print("TEST COMPLETED.")

except Exception as e:
    print(f"\nFATAL ERROR DURING MODEL LOAD/INFERENCE: {e}")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"VRAM at failure: Allocated={alloc:.2f} MiB, Peak={peak:.2f} MiB")
    sys.exit(1)
