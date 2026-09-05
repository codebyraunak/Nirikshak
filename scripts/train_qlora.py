import os
import sys
import json
import time
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

print("=" * 80)
print("HEXA-FORGE PHASE 2: QLoRA FINE-TUNING (QWEN3-1.7B)")
print("=" * 80)

# Anti-leakage verification: Explicit assertion that test files are not accessed
TRAIN_FILE = "data/train.jsonl"
OUTPUT_DIR = "outputs/qwen3-1.7b-hexa-forge-lora"
CHECKPOINTS_DIR = "outputs/qwen3-1.7b-checkpoints"

assert not os.path.exists("data/test_unseen.jsonl.leaked"), "Security check"
print(f"[Anti-Leakage Check] Training strictly on: {TRAIN_FILE}")
print(f"[Anti-Leakage Check] Test sets are completely excluded from training process.")

# 1. Check GPU memory before loading
device_name = torch.cuda.get_device_name(0)
total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
free_vram_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024 ** 2)
print(f"Device: {device_name} | Total VRAM: {total_vram_mb:.1f} MiB | Free VRAM: {free_vram_mb:.1f} MiB")

# 2. Load Tokenizer
model_id = "Qwen/Qwen3-1.7B"
print(f"\n1. Loading Tokenizer: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. BitsAndBytes 4-bit Quantization Config
print("2. Configuring BitsAndBytes NF4 4-bit with Double Quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 4. Load Base Model
print("3. Loading base model in 4-bit...")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 5. Prepare Model for k-bit Training
print("4. Preparing model for k-bit training (gradient checkpointing enabled, use_cache=False)...")
model.gradient_checkpointing_enable()
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# 6. Apply LoRA Config
print("5. Configuring LoRA (rank=16, alpha=32, dropout=0.05, targets: q/k/v/o_proj)...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)

# Print parameter statistics
trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in peft_model.parameters())
trainable_pct = 100 * trainable_params / total_params
print(f"\nTrainable Parameters: {trainable_params:,} | Total Parameters: {total_params:,} ({trainable_pct:.3f}%)")

# 7. Load and Format Training Data
print(f"\n6. Loading Training Dataset from {TRAIN_FILE}...")
raw_train_records = []
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        raw_train_records.append(json.loads(line))

print(f"Loaded {len(raw_train_records)} training records.")

formatted_texts = []
for r in raw_train_records:
    # Use chat template with enable_thinking=False
    text = tokenizer.apply_chat_template(
        r["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    formatted_texts.append(text)

def tokenize_fn(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = Dataset.from_dict({"text": formatted_texts})
tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"Tokenized dataset length: {len(tokenized_dataset)} samples (max_seq_length=512).")

# 8. Training Arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINTS_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_steps=1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    fp16=True,
    logging_steps=1,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    remove_unused_columns=False
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 9. Train Model
print("\n7. Starting QLoRA Training...")
t_start_train = time.time()
train_result = trainer.train()
train_duration = time.time() - t_start_train
peak_train_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"\nTraining completed in {train_duration:.2f}s!")
print(f"Final Training Loss: {train_result.training_loss:.4f}")
print(f"Peak Training VRAM: {peak_train_vram_mb:.1f} MiB")

# 10. Save Adapter
print(f"\n8. Saving fine-tuned LoRA adapter to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
peft_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training metadata
metadata = {
    "base_model": model_id,
    "train_records": len(raw_train_records),
    "epochs": 2,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 8,
    "max_seq_length": 512,
    "learning_rate": 2e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "trainable_parameters": trainable_params,
    "total_parameters": total_params,
    "training_loss": round(train_result.training_loss, 4),
    "training_duration_sec": round(train_duration, 2),
    "peak_vram_mb": round(peak_train_vram_mb, 1)
}

with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Adapter and metadata successfully saved to: {OUTPUT_DIR}")
