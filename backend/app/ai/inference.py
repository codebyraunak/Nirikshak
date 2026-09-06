from pathlib import Path
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen3-1.7B"

ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "qwen3-1.7b-hexa-forge-lora"
)


class QwenInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return

        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(
                f"Qwen adapter not found at: {ADAPTER_PATH}"
            )

        print("Loading Qwen tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ADAPTER_PATH),
            trust_remote_code=True,
        )

        print("Loading Qwen base model...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        print("Loading HEXA-FORGE LoRA adapter...")

        self.model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_PATH),
        )

        self.model.eval()

        self.loaded = True

        print("Qwen3-1.7B + HEXA-FORGE LoRA loaded successfully.")

    def interpret(
        self,
        vendor: str,
        platform: str,
        section: str,
        previous_lines: list[str],
        current_line: str,
        next_lines: list[str],
    ) -> dict:

        self.load()

        previous_context = "\n".join(previous_lines)
        next_context = "\n".join(next_lines)

        system_prompt = """
You are the configuration interpretation engine for HEXA-FORGE,
an AI-augmented network configuration security auditor.

Your job is NOT to decide whether a configuration is compliant.

Your job is only to interpret network configuration syntax
and map it to a vendor-neutral security control.

Supported canonical controls:

SSH_VERSION
TELNET_ENABLED
HTTP_MANAGEMENT
SESSION_TIMEOUT
AAA_ENABLED
UNKNOWN

Supported categories:

session_security
management_plane
access_control
none

Rules:

1. Never invent configuration evidence.
2. Use only the supplied configuration context.
3. Preserve the actual configured value.
4. Return exactly one canonical security control.
5. If the configuration is unrelated to network security, use UNKNOWN.
6. Confidence must be between 0 and 1.
7. If confidence is below 0.85, set requires_human_validation to true.
8. Do not calculate compliance.
9. Do not claim CIS, NIST, STIG, or ISO compliance.
10. Do not generate remediation commands.
11. Evidence must reproduce the exact supplied configuration line.
12. Return JSON only.
"""

        user_prompt = f"""
Vendor:
{vendor}

Platform:
{platform}

Configuration section:
{section}

Previous configuration lines:
{previous_context}

Unknown configuration line:
{current_line}

Following configuration lines:
{next_context}

Interpret the unknown configuration line.

Return exactly:

{{
  "security_control": "CONTROL_ID",
  "value": "actual value",
  "security_category": "category",
  "security_meaning": "plain English explanation",
  "confidence": 0.0,
  "evidence": [
    {{
      "line": 0,
      "text": "exact configuration line"
    }}
  ],
  "requires_human_validation": true
}}
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        response_text = self._clean_response(response_text)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Qwen returned invalid JSON: {response_text}"
            ) from exc

        return result

    @staticmethod
    def _clean_response(text: str) -> str:

        # Remove Qwen thinking blocks if present.
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL,
        ).strip()

        # Remove markdown JSON fences.
        text = re.sub(
            r"^```json\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"^```\s*",
            "",
            text,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        ).strip()

        return text


qwen = QwenInference()