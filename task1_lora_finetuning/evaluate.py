"""
evaluate.py
-----------
Standalone evaluation script for the fine-tuned LoRA model.

Usage (after training):
    python evaluate.py \
        --base_model Qwen/Qwen2.5-0.5B-Instruct \
        --lora_adapter ./lora_adapter \
        --test_file ./data/test.json \
        --output_file ./results/eval_results.json

Metrics:
    - Exact Match Accuracy  : output JSON string identical to target
    - Key Accuracy          : fraction of keys correctly parsed
    - Value Accuracy        : fraction of key-value pairs correctly parsed
    - Valid JSON Rate        : fraction of outputs that are parseable JSON
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def exact_match(pred: str, gold: str) -> bool:
    """Exact string match after normalising whitespace."""
    try:
        # Normalise by re-serialising with same settings
        pred_obj = json.loads(pred)
        gold_obj = json.loads(gold)
        return json.dumps(pred_obj, sort_keys=True) == json.dumps(gold_obj, sort_keys=True)
    except (json.JSONDecodeError, ValueError):
        return pred.strip() == gold.strip()


def key_accuracy(pred: str, gold: str) -> float:
    """Fraction of gold keys present in predicted JSON."""
    try:
        pred_obj = json.loads(pred)
        gold_obj = json.loads(gold)
        if not gold_obj:
            return 1.0
        pred_keys = set(pred_obj.keys())
        gold_keys = set(gold_obj.keys())
        return len(pred_keys & gold_keys) / len(gold_keys)
    except (json.JSONDecodeError, ValueError):
        return 0.0


def value_accuracy(pred: str, gold: str) -> float:
    """Fraction of key-value pairs that match (key present AND value equal)."""
    try:
        pred_obj = json.loads(pred)
        gold_obj = json.loads(gold)
        if not gold_obj:
            return 1.0
        matches = sum(
            1 for k, v in gold_obj.items()
            if k in pred_obj and str(pred_obj[k]).lower() == str(v).lower()
        )
        return matches / len(gold_obj)
    except (json.JSONDecodeError, ValueError):
        return 0.0


def compute_metrics(predictions: list[str], gold_outputs: list[str]) -> dict:
    """Compute all metrics across the test set."""
    n = len(predictions)
    assert n == len(gold_outputs), "Length mismatch between predictions and gold outputs"

    em_scores = [exact_match(p, g) for p, g in zip(predictions, gold_outputs)]
    key_scores = [key_accuracy(p, g) for p, g in zip(predictions, gold_outputs)]
    val_scores = [value_accuracy(p, g) for p, g in zip(predictions, gold_outputs)]
    valid_json = [is_valid_json(p) for p in predictions]

    return {
        "num_examples": n,
        "exact_match_accuracy": sum(em_scores) / n,
        "key_accuracy": sum(key_scores) / n,
        "value_accuracy": sum(val_scores) / n,
        "valid_json_rate": sum(valid_json) / n,
        "per_example": [
            {
                "exact_match": em,
                "key_accuracy": ka,
                "value_accuracy": va,
                "valid_json": vj,
            }
            for em, ka, va, vj in zip(em_scores, key_scores, val_scores, valid_json)
        ],
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(base_model: str, lora_adapter: str | None, device: str):
    """Load tokenizer and model (optionally with LoRA adapter)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading tokenizer from: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if device != "cpu":
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            quant_config = None

    print(f"Loading base model from: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )

    if lora_adapter and os.path.isdir(lora_adapter):
        print(f"Loading LoRA adapter from: {lora_adapter}")
        model = PeftModel.from_pretrained(model, lora_adapter)
        model = model.merge_and_unload()
        print("LoRA adapter merged successfully.")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a JSON extraction assistant. Convert the user's plain-English request "
    "into a compact JSON object. Output ONLY valid JSON, no explanations, no markdown."
)


def build_prompt(tokenizer, input_text: str) -> str:
    """Build the chat-formatted prompt for a single example."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"System: {SYSTEM_PROMPT}\nUser: {input_text}\nAssistant:"


def generate_prediction(
    model, tokenizer, input_text: str, device: str, max_new_tokens: int = 128
) -> str:
    """Run inference and extract the JSON portion of the output."""
    prompt = build_prompt(tokenizer, input_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Try to extract a JSON object from the output
    start = decoded.find("{")
    end = decoded.rfind("}") + 1
    if start != -1 and end > start:
        return decoded[start:end]
    return decoded


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load test data
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Test examples: {len(test_data)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.lora_adapter, device
    )

    predictions = []
    gold_outputs = []
    input_texts = []

    print("\nRunning inference...")
    t0 = time.time()
    for ex in tqdm(test_data):
        pred = generate_prediction(model, tokenizer, ex["input"], device)
        predictions.append(pred)
        gold_outputs.append(ex["output"])
        input_texts.append(ex["input"])
    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.1f}s ({elapsed/len(test_data):.2f}s/example)\n")

    # Compute metrics
    metrics = compute_metrics(predictions, gold_outputs)

    # Print summary
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Examples evaluated : {metrics['num_examples']}")
    print(f"  Valid JSON Rate    : {metrics['valid_json_rate']:.2%}")
    print(f"  Exact Match Acc.   : {metrics['exact_match_accuracy']:.2%}")
    print(f"  Key Accuracy       : {metrics['key_accuracy']:.2%}")
    print(f"  Value Accuracy     : {metrics['value_accuracy']:.2%}")
    print("=" * 60)

    # Show failures
    failures = [
        i for i, ex in enumerate(metrics["per_example"]) if not ex["exact_match"]
    ]
    print(f"\nFailed examples: {len(failures)} / {metrics['num_examples']}")
    if failures and args.show_failures > 0:
        print(f"\n--- Sample failures (first {args.show_failures}) ---")
        for i in failures[: args.show_failures]:
            print(f"\n[{i}] Input : {input_texts[i]}")
            print(f"     Gold  : {gold_outputs[i]}")
            print(f"     Pred  : {predictions[i]}")
            print(f"     Key Acc: {metrics['per_example'][i]['key_accuracy']:.2%} | "
                  f"Val Acc: {metrics['per_example'][i]['value_accuracy']:.2%}")

    # Save results
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        result = {
            "model": args.base_model,
            "lora_adapter": args.lora_adapter,
            "metrics": {k: v for k, v in metrics.items() if k != "per_example"},
            "predictions": [
                {"input": inp, "gold": gold, "pred": pred, **stats}
                for inp, gold, pred, stats in zip(
                    input_texts, gold_outputs, predictions, metrics["per_example"]
                )
            ],
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--lora_adapter", default=None,
                        help="Path to LoRA adapter directory (None = evaluate base model)")
    parser.add_argument("--test_file", default="./data/test.json",
                        help="Path to test.json")
    parser.add_argument("--output_file", default="./results/eval_results.json",
                        help="Where to save evaluation results JSON")
    parser.add_argument("--show_failures", type=int, default=5,
                        help="Number of failure cases to print")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.test_file):
        print(f"ERROR: Test file not found: {args.test_file}", file=sys.stderr)
        sys.exit(1)

    run_evaluation(args)


if __name__ == "__main__":
    main()
