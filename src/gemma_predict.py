"""Zero-shot CWE inference using CodeGemma-2B with 4-bit quantization.

No fine-tuning required. Uses structured prompting + beam search to produce
top-K CWE predictions with approximate confidence scores.
"""

import re
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocess import preprocess

MODEL_ID = "google/codegemma-2b"

# All 118 CWE IDs present in the Juliet Test Suite label map
_VALID_CWE_IDS = {
    "15", "23", "36", "78", "90", "114", "121", "122", "123", "124",
    "126", "127", "134", "176", "188", "190", "191", "194", "195", "196",
    "197", "222", "223", "226", "242", "244", "247", "252", "253", "256",
    "259", "272", "273", "284", "319", "321", "325", "327", "328", "338",
    "364", "366", "367", "369", "377", "390", "391", "396", "397", "398",
    "400", "401", "404", "415", "416", "426", "427", "440", "457", "459",
    "464", "467", "468", "469", "475", "476", "478", "479", "480", "481",
    "482", "483", "484", "500", "506", "510", "511", "526", "534", "535",
    "546", "561", "562", "563", "570", "571", "587", "588", "590", "591",
    "605", "606", "615", "617", "620", "665", "666", "667", "672", "674",
    "675", "676", "680", "681", "685", "688", "690", "758", "761", "762",
    "773", "775", "780", "785", "789", "832", "835", "843",
}

_CWE_LIST_STR = ", ".join(sorted(_VALID_CWE_IDS, key=int))

_PROMPT_TEMPLATE = """\
You are a C/C++ security expert. Analyze the code below and identify the CWE vulnerability number it contains.

Valid CWE numbers: {cwe_list}

Respond with ONLY the CWE number (e.g. 121). Do not explain.

Code:
```c
{code}
```

CWE number: """


def _build_prompt(code: str, max_code_chars: int = 800) -> str:
    """Build the zero-shot prompt, truncating code to fit context."""
    code = preprocess(code)
    if len(code) > max_code_chars:
        code = code[:max_code_chars] + "\n// ... (truncated)"
    return _PROMPT_TEMPLATE.format(cwe_list=_CWE_LIST_STR, code=code)


def load_gemma_model():
    """Load CodeGemma-2B with 4-bit NF4 quantization.

    Falls back to float16 (no quantization) if bitsandbytes is unavailable
    or CUDA is not present.

    Returns:
        (model, tokenizer) tuple
    """
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)

    use_4bit = torch.cuda.is_available()
    if use_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token,
            )
        except Exception:
            use_4bit = False

    if not use_4bit:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=hf_token,
        ).to(device)

    model.eval()
    return model, tokenizer


def predict_code_gemma(
    code: str,
    model,
    tokenizer,
    label_map: dict,
    top_k: int = 5,
    max_new_tokens: int = 8,
) -> list[dict]:
    """Zero-shot CWE prediction using CodeGemma beam search.

    Args:
        code: Raw C/C++ source code string.
        model: Loaded CodeGemma causal LM.
        tokenizer: Corresponding tokenizer.
        label_map: Dict mapping CWE ID string → class index (for validation).
        top_k: Number of top predictions to return.
        max_new_tokens: Max tokens to generate per beam (short — just a number).

    Returns:
        List of {"cwe": "CWE-X", "confidence": float} dicts, sorted descending.
    """
    prompt = _build_prompt(code)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(top_k * 2, 10),   # over-generate then filter
            num_return_sequences=max(top_k * 2, 10),
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Decode generated tokens only (skip prompt)
    sequences = outputs.sequences[:, input_len:]
    # sequences_scores: shape (num_return_sequences,) — log-prob of each beam
    scores = outputs.sequences_scores  # log probs

    # Parse CWE numbers from each beam
    seen: set[str] = set()
    candidates: list[tuple[str, float]] = []
    for seq, score in zip(sequences, scores):
        text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        # Extract first integer in the output
        match = re.search(r"\b(\d+)\b", text)
        if not match:
            continue
        cwe_num = match.group(1)
        cwe_id = f"CWE-{cwe_num}"
        # Must be a known CWE from our label map
        if cwe_num not in _VALID_CWE_IDS:
            continue
        if cwe_id in seen:
            continue
        seen.add(cwe_id)
        candidates.append((cwe_id, score.item()))
        if len(candidates) >= top_k:
            break

    if not candidates:
        # Graceful fallback: return top-k valid CWEs with uniform low confidence
        fallback = [
            {"cwe": f"CWE-{c}", "confidence": 1.0 / top_k}
            for c in sorted(_VALID_CWE_IDS, key=int)[:top_k]
        ]
        return fallback

    # Convert log probs → normalised confidences via softmax
    raw_scores = torch.tensor([s for _, s in candidates])
    confidences = torch.softmax(raw_scores, dim=0).tolist()

    results = [
        {"cwe": cwe_id, "confidence": conf}
        for (cwe_id, _), conf in zip(candidates, confidences)
    ]
    return sorted(results, key=lambda x: x["confidence"], reverse=True)


# ---------------------------------------------------------------------------
# Generic zero-shot loader — supports any causal LM with 4-bit quantization
# ---------------------------------------------------------------------------

_CHAT_PROMPT_TEMPLATE = """\
You are a C/C++ security expert. Analyze the following code and identify the CWE vulnerability number it contains.

Valid CWE numbers: {cwe_list}

Respond with ONLY the CWE number (e.g. 121). Do not explain.

Code:
```c
{code}
```"""


def _is_instruction_tuned(model_id: str) -> bool:
    """Return True if model uses chat/instruction format (-it suffix)."""
    lower = model_id.lower()
    return lower.endswith("-it") or "-it-" in lower or lower.endswith("-it")


def _build_prompt_for_model(code: str, model_id: str, tokenizer, max_code_chars: int = 800) -> str:
    """Build a completion prompt or chat prompt depending on model type."""
    code = preprocess(code)
    if len(code) > max_code_chars:
        code = code[:max_code_chars] + "\n// ... (truncated)"
    if _is_instruction_tuned(model_id):
        messages = [{"role": "user", "content": _CHAT_PROMPT_TEMPLATE.format(
            cwe_list=_CWE_LIST_STR, code=code)}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _PROMPT_TEMPLATE.format(cwe_list=_CWE_LIST_STR, code=code)


def load_zero_shot_model(model_id: str):
    """Load any causal LM with 4-bit NF4 quantization (or float16 fallback).

    For models with "bnb-4bit" in the ID (pre-quantized), loads weights
    directly without re-applying BitsAndBytesConfig.
    Otherwise applies on-the-fly NF4 quantization via bitsandbytes.

    Returns:
        (model, tokenizer) tuple
    """
    # Monkey-patch bitsandbytes to accept _is_hf_initialized kwarg
    # (transformers 5.x passes it, bitsandbytes 0.49.x doesn't expect it)
    try:
        import bitsandbytes as bnb
        _orig_params4bit_new = bnb.nn.Params4bit.__new__
        def _patched_params4bit_new(cls, *args, **kwargs):
            kwargs.pop("_is_hf_initialized", None)
            return _orig_params4bit_new(cls, *args, **kwargs)
        bnb.nn.Params4bit.__new__ = _patched_params4bit_new
    except Exception:
        pass

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    is_prebuilt = "bnb-4bit" in model_id

    if is_prebuilt:
        # Pre-quantized weights — load directly; bitsandbytes handles inference
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            token=hf_token,
        )
        print(f"[ZeroShot] Loaded pre-built 4-bit model: {model_id}")
    elif torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token,
            )
            print(f"[ZeroShot] Loaded with 4-bit NF4 quantization: {model_id}")
        except Exception as e:
            print(f"[ZeroShot] 4-bit load failed ({e}), falling back to float16.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map="auto", token=hf_token,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, token=hf_token,
        )
        print(f"[ZeroShot] Loaded in float32 (CPU): {model_id}")

    model.eval()
    return model, tokenizer


def predict_code_zero_shot(
    code: str,
    model,
    tokenizer,
    model_id: str,
    label_map: dict,
    top_k: int = 5,
    max_new_tokens: int = 8,
    num_beams: int | None = None,
) -> list[dict]:
    """Zero-shot CWE prediction for any causal LM.

    Uses chat template for instruction-tuned models (-it suffix),
    raw completion prompt otherwise.

    Returns:
        List of {"cwe": "CWE-X", "confidence": float} dicts, sorted descending.
    """
    if num_beams is None:
        num_beams = max(top_k * 2, 10)
    num_return = min(num_beams, max(top_k * 2, 10)) if num_beams > 1 else 1
    prompt = _build_prompt_for_model(code, model_id, tokenizer)
    # With device_map="auto", model.device may not reflect actual compute device.
    # Determine input device from the first parameter instead.
    _device = next(p.device for p in model.parameters() if p.device.type != "meta")
    inputs = tokenizer(prompt, return_tensors="pt").to(_device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        early_stopping=True if num_beams > 1 else False,
        return_dict_in_generate=True,
    )
    if num_beams > 1:
        gen_kwargs.update(
            num_beams=num_beams,
            num_return_sequences=num_return,
            output_scores=True,
        )
    else:
        gen_kwargs.update(
            do_sample=False,
            num_return_sequences=1,
        )

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    sequences = outputs.sequences[:, input_len:]

    if num_beams > 1:
        scores = outputs.sequences_scores.float().cpu()
    else:
        scores = torch.zeros(sequences.shape[0])

    seen: set[str] = set()
    candidates: list[tuple[str, float]] = []
    for seq, score in zip(sequences, scores):
        text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        match = re.search(r"\b(\d+)\b", text)
        if not match:
            continue
        cwe_num = match.group(1)
        cwe_id = f"CWE-{cwe_num}"
        if cwe_num not in _VALID_CWE_IDS:
            continue
        if cwe_id in seen:
            continue
        seen.add(cwe_id)
        candidates.append((cwe_id, score.item()))
        if len(candidates) >= top_k:
            break

    if not candidates:
        return [
            {"cwe": f"CWE-{c}", "confidence": 1.0 / top_k}
            for c in sorted(_VALID_CWE_IDS, key=int)[:top_k]
        ]

    raw_scores = torch.tensor([s for _, s in candidates])
    confidences = torch.softmax(raw_scores, dim=0).tolist()

    return sorted(
        [{"cwe": cwe_id, "confidence": conf}
         for (cwe_id, _), conf in zip(candidates, confidences)],
        key=lambda x: x["confidence"], reverse=True,
    )
