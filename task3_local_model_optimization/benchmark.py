"""
benchmark.py – Main benchmark script for Task 3: Local Model Optimisation.

Benchmarks two Ollama models across 10 standardised prompts and measures:
  • Tokens per second  (eval_count / eval_duration from Ollama JSON response)
  • GPU / CPU memory   (Ollama /api/ps endpoint + nvidia-smi fallback)
  • Wall-clock latency per prompt

Results are saved to benchmark_results.json and a formatted table is printed.

Usage:
    python benchmark.py [--model1 NAME] [--model2 NAME] [--out FILE] [--timeout N]

Defaults:
    model1  = llama3.2:1b          (default quantisation – usually Q4_K_M)
    model2  = llama3.2:1b:q4_0     (explicit 4-bit Q4_0 quantisation)
    out     = benchmark_results.json
    timeout = 120  (seconds per request)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import psutil
import requests
from tabulate import tabulate

# ── Import the shared prompt list ─────────────────────────────────────────────
try:
    from test_prompts_loader import TEST_PROMPTS          # thin re-export shim
except ImportError:
    from importlib.util import spec_from_file_location, module_from_spec
    import pathlib
    _spec = spec_from_file_location(
        "test_prompts_loader",
        pathlib.Path(__file__).parent / "10_test_prompts.py",
    )
    _mod = module_from_spec(_spec)           # type: ignore[arg-type]
    _spec.loader.exec_module(_mod)           # type: ignore[union-attr]
    TEST_PROMPTS = _mod.TEST_PROMPTS

# ── Constants ─────────────────────────────────────────────────────────────────
OLLAMA_BASE       = "http://localhost:11434"
GENERATE_ENDPOINT = f"{OLLAMA_BASE}/api/generate"
PS_ENDPOINT       = f"{OLLAMA_BASE}/api/ps"
TAGS_ENDPOINT     = f"{OLLAMA_BASE}/api/tags"
PULL_ENDPOINT     = f"{OLLAMA_BASE}/api/pull"
NS_PER_SEC        = 1_000_000_000          # Ollama reports durations in nanoseconds

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama_running(timeout: int = 5) -> bool:
    """Return True if the Ollama server responds on its health endpoint."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def list_local_models() -> list[str]:
    """Return names of models already pulled locally."""
    try:
        r = requests.get(TAGS_ENDPOINT, timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def pull_model(model: str) -> bool:
    """
    Stream-pull a model via Ollama API.
    Returns True on success, False on failure.
    """
    print(f"  Pulling model '{model}' (this may take a while)…")
    try:
        with requests.post(
            PULL_ENDPOINT,
            json={"name": model, "stream": True},
            stream=True,
            timeout=600,
        ) as resp:
            resp.raise_for_status()
            last_status = ""
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if status != last_status:
                        print(f"    [{status}]")
                        last_status = status
                    if data.get("status") == "success":
                        print(f"  ✔ Model '{model}' ready.")
                        return True
        return True
    except Exception as exc:
        print(f"  ✘ Failed to pull '{model}': {exc}")
        return False


def ensure_model_available(model: str) -> bool:
    """Pull the model if it is not already present. Return True if available."""
    local = list_local_models()
    # Ollama tag matching: treat "model:latest" and "model" as equivalent
    normalised = [m.split(":")[0] if m.endswith(":latest") else m for m in local]
    if model in local or model in normalised:
        print(f"  ✔ Model '{model}' already present locally.")
        return True
    return pull_model(model)


# ─────────────────────────────────────────────────────────────────────────────
# Memory measurement
# ─────────────────────────────────────────────────────────────────────────────

def get_memory_via_ollama_ps() -> dict[str, Any]:
    """
    Query Ollama /api/ps for currently loaded models and their memory.
    Returns a dict  { model_name: size_vram_bytes }  (may be empty).
    """
    try:
        r = requests.get(PS_ENDPOINT, timeout=10)
        r.raise_for_status()
        result: dict[str, int] = {}
        for m in r.json().get("models", []):
            name = m.get("name", "unknown")
            size = m.get("size_vram", m.get("size", 0))   # bytes
            result[name] = size
        return result
    except Exception:
        return {}


def get_gpu_memory_mb_nvidia_smi() -> dict[str, int]:
    """
    Fallback: query nvidia-smi for per-GPU used memory.
    Returns {gpu_index: used_mib}.
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            text=True,
        )
        result: dict[str, int] = {}
        for line in output.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                idx, used = parts
                result[f"gpu{idx.strip()}"] = int(used.strip())
        return result
    except Exception:
        return {}


def get_process_rss_mb() -> float:
    """Return RSS of the current Python process in MiB (CPU-side fallback)."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 2)


def snapshot_memory(model_name: str) -> dict[str, Any]:
    """
    Take a memory snapshot combining all available sources.
    Priority: Ollama /api/ps  >  nvidia-smi  >  psutil RSS
    """
    ollama_ps    = get_memory_via_ollama_ps()
    nvidia       = get_gpu_memory_mb_nvidia_smi()
    cpu_rss_mb   = get_process_rss_mb()

    # Try to find this model's VRAM usage from Ollama /api/ps
    vram_bytes = None
    for name, size in ollama_ps.items():
        if model_name.split(":")[0] in name:
            vram_bytes = size
            break
    if vram_bytes is None and ollama_ps:
        # Fall back to any loaded model's size (useful when only one is loaded)
        vram_bytes = next(iter(ollama_ps.values()), None)

    return {
        "ollama_ps_vram_bytes"  : vram_bytes,
        "ollama_ps_vram_mb"     : round(vram_bytes / (1024 ** 2), 1) if vram_bytes else None,
        "nvidia_smi_used_mib"   : nvidia,
        "python_rss_mb"         : round(cpu_rss_mb, 1),
        "source"                : (
            "ollama_ps"  if vram_bytes  else
            "nvidia_smi" if nvidia      else
            "psutil_rss"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_single_prompt(
    model: str,
    prompt_obj: dict,
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Send one prompt to Ollama and return a detailed result dict.

    Key metrics extracted from Ollama's response JSON:
      eval_count    – number of tokens generated
      eval_duration – time (ns) spent generating those tokens
      → tokens_per_second = eval_count / (eval_duration / 1e9)
    """
    prompt_text = prompt_obj["prompt"]
    t_wall_start = time.perf_counter()

    try:
        response = requests.post(
            GENERATE_ENDPOINT,
            json={
                "model"  : model,
                "prompt" : prompt_text,
                "stream" : False,          # get a single JSON blob back
                "options": {
                    "temperature": 0.1,    # low temp → reproducible outputs
                    "num_predict": 256,    # cap output length for speed
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        t_wall_end = time.perf_counter()

        data: dict = response.json()

        eval_count      = data.get("eval_count",    0)
        eval_duration   = data.get("eval_duration", 1)   # nanoseconds
        prompt_eval_count    = data.get("prompt_eval_count",    0)
        prompt_eval_duration = data.get("prompt_eval_duration", 1)

        tokens_per_sec = (
            eval_count / (eval_duration / NS_PER_SEC)
            if eval_duration > 0 and eval_count > 0
            else 0.0
        )
        prompt_tps = (
            prompt_eval_count / (prompt_eval_duration / NS_PER_SEC)
            if prompt_eval_duration > 0 and prompt_eval_count > 0
            else 0.0
        )

        return {
            "prompt_id"          : prompt_obj["id"],
            "category"           : prompt_obj["category"],
            "prompt"             : prompt_text,
            "response"           : data.get("response", ""),
            "tokens_generated"   : eval_count,
            "tokens_per_second"  : round(tokens_per_sec, 2),
            "prompt_tokens"      : prompt_eval_count,
            "prompt_tps"         : round(prompt_tps, 2),
            "eval_duration_s"    : round(eval_duration / NS_PER_SEC, 3),
            "total_duration_s"   : round(data.get("total_duration", 0) / NS_PER_SEC, 3),
            "wall_clock_s"       : round(t_wall_end - t_wall_start, 3),
            "done"               : data.get("done", False),
            "error"              : None,
        }

    except requests.exceptions.Timeout:
        t_wall_end = time.perf_counter()
        return {
            "prompt_id"         : prompt_obj["id"],
            "category"          : prompt_obj["category"],
            "prompt"            : prompt_text,
            "response"          : "",
            "tokens_generated"  : 0,
            "tokens_per_second" : 0.0,
            "prompt_tokens"     : 0,
            "prompt_tps"        : 0.0,
            "eval_duration_s"   : 0.0,
            "total_duration_s"  : 0.0,
            "wall_clock_s"      : round(t_wall_end - t_wall_start, 3),
            "done"              : False,
            "error"             : f"Timeout after {timeout}s",
        }
    except Exception as exc:
        return {
            "prompt_id"         : prompt_obj["id"],
            "category"          : prompt_obj["category"],
            "prompt"            : prompt_text,
            "response"          : "",
            "tokens_generated"  : 0,
            "tokens_per_second" : 0.0,
            "prompt_tokens"     : 0,
            "prompt_tps"        : 0.0,
            "eval_duration_s"   : 0.0,
            "total_duration_s"  : 0.0,
            "wall_clock_s"      : 0.0,
            "done"              : False,
            "error"             : str(exc),
        }


def benchmark_model(
    model: str,
    prompts: list[dict],
    timeout: int = 120,
    warmup: bool = True,
) -> dict[str, Any]:
    """
    Run the full benchmark for a single model.
    Returns a structured result dict ready for JSON serialisation.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {model}")
    print(f"{'='*60}")

    # ── Optional warmup (loads model into VRAM, avoids cold-start bias) ───────
    if warmup:
        print("  Running warmup prompt…")
        warmup_result = run_single_prompt(
            model,
            {"id": 0, "category": "warmup", "prompt": "Say hello."},
            timeout=timeout,
        )
        if warmup_result["error"]:
            print(f"  ⚠ Warmup failed: {warmup_result['error']}")
        else:
            print(f"  ✔ Warmup done  ({warmup_result['tokens_per_second']} tok/s)")

    # ── Memory snapshot after warmup (model now loaded in VRAM) ──────────────
    mem_snapshot = snapshot_memory(model)
    print(
        f"  Memory snapshot: "
        f"{mem_snapshot.get('ollama_ps_vram_mb', 'N/A')} MB VRAM  "
        f"(source: {mem_snapshot['source']})"
    )

    # ── Run all 10 prompts ────────────────────────────────────────────────────
    results: list[dict] = []
    for i, prompt_obj in enumerate(prompts, 1):
        print(f"  [{i:02d}/10] {prompt_obj['category']:18s} ", end="", flush=True)
        result = run_single_prompt(model, prompt_obj, timeout=timeout)
        results.append(result)

        if result["error"]:
            print(f"ERROR – {result['error']}")
        else:
            print(
                f"{result['tokens_per_second']:6.1f} tok/s  "
                f"{result['tokens_generated']:4d} tokens  "
                f"{result['wall_clock_s']:6.2f}s"
            )

    # ── Aggregate statistics ──────────────────────────────────────────────────
    valid = [r for r in results if not r["error"] and r["tokens_generated"] > 0]
    avg_tps       = sum(r["tokens_per_second"] for r in valid) / len(valid) if valid else 0
    avg_latency   = sum(r["wall_clock_s"]     for r in valid) / len(valid) if valid else 0
    total_tokens  = sum(r["tokens_generated"] for r in valid)
    errors        = len(results) - len(valid)

    summary = {
        "model"              : model,
        "timestamp"          : datetime.now(timezone.utc).isoformat(),
        "total_prompts"      : len(prompts),
        "successful_prompts" : len(valid),
        "errors"             : errors,
        "avg_tokens_per_sec" : round(avg_tps, 2),
        "avg_wall_clock_s"   : round(avg_latency, 3),
        "total_tokens_generated": total_tokens,
        "memory"             : mem_snapshot,
        "prompt_results"     : results,
    }

    print(f"\n  → Avg {avg_tps:.1f} tok/s   |   {errors} error(s)")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Output / display
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: list[dict]) -> None:
    """Print a human-readable side-by-side comparison of all benchmarked models."""
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    # Per-model summary table
    summary_rows = []
    for r in results:
        mem = r["memory"]
        vram = mem.get("ollama_ps_vram_mb")
        vram_str = f"{vram:.0f} MB" if vram else "N/A"
        summary_rows.append([
            r["model"],
            f"{r['avg_tokens_per_sec']:.1f}",
            vram_str,
            f"{r['avg_wall_clock_s']:.2f}s",
            r["successful_prompts"],
            r["errors"],
        ])

    print(tabulate(
        summary_rows,
        headers=["Model", "Avg tok/s", "VRAM Used", "Avg Latency", "OK", "Errors"],
        tablefmt="rounded_outline",
    ))

    # Per-prompt comparison
    if len(results) == 2:
        m1, m2 = results[0], results[1]
        rows = []
        for p1, p2 in zip(m1["prompt_results"], m2["prompt_results"]):
            rows.append([
                p1["prompt_id"],
                p1["category"],
                f"{p1['tokens_per_second']:.1f}" if not p1["error"] else "ERR",
                f"{p2['tokens_per_second']:.1f}" if not p2["error"] else "ERR",
                f"{p1['tokens_generated']}"       if not p1["error"] else "–",
                f"{p2['tokens_generated']}"       if not p2["error"] else "–",
            ])

        print(f"\nPer-prompt breakdown: {m1['model']}  vs  {m2['model']}")
        print(tabulate(
            rows,
            headers=[
                "#", "Category",
                f"tok/s\n{m1['model']}", f"tok/s\n{m2['model']}",
                f"tokens\n{m1['model']}", f"tokens\n{m2['model']}",
            ],
            tablefmt="rounded_outline",
        ))

        # Speedup
        if m1["avg_tokens_per_sec"] and m2["avg_tokens_per_sec"]:
            faster = max(results, key=lambda r: r["avg_tokens_per_sec"])
            slower = min(results, key=lambda r: r["avg_tokens_per_sec"])
            ratio  = faster["avg_tokens_per_sec"] / slower["avg_tokens_per_sec"]
            print(f"\n  🏆  {faster['model']} is {ratio:.2f}× faster than {slower['model']}")

        v1 = m1["memory"].get("ollama_ps_vram_mb")
        v2 = m2["memory"].get("ollama_ps_vram_mb")
        if v1 and v2 and v1 != v2:
            smaller = min(v1, v2)
            larger  = max(v1, v2)
            saving  = (1 - smaller / larger) * 100
            print(f"  💾  Memory saving: {saving:.1f}% less VRAM for the quantised model")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark two Ollama models and compare performance."
    )
    p.add_argument("--model1",   default="llama3.2:1b",
                   help="First model tag  (default: llama3.2:1b)")
    p.add_argument("--model2",   default="llama3.2:1b:q4_0",
                   help="Second model tag (default: llama3.2:1b:q4_0)")
    p.add_argument("--out",      default="benchmark_results.json",
                   help="Output JSON file (default: benchmark_results.json)")
    p.add_argument("--timeout",  type=int, default=120,
                   help="Per-request timeout in seconds (default: 120)")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip warmup prompt")
    p.add_argument("--no-pull",   action="store_true",
                   help="Do not attempt to pull missing models")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Task 3 – Local Model Optimisation Benchmark            ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Pre-flight: Ollama running? ───────────────────────────────────────────
    if not check_ollama_running():
        print("✘ ERROR: Ollama is not running on http://localhost:11434")
        print("  Start Ollama with:  ollama serve")
        print("  Then re-run this script.")
        sys.exit(1)
    print("✔ Ollama is running.\n")

    models = [args.model1, args.model2]

    # ── Ensure models are available ───────────────────────────────────────────
    if not args.no_pull:
        print("Checking model availability…")
        for model in models:
            if not ensure_model_available(model):
                print(f"✘ Could not obtain model '{model}'. Exiting.")
                sys.exit(1)
        print()

    # ── Run benchmarks ────────────────────────────────────────────────────────
    all_results: list[dict] = []
    for model in models:
        result = benchmark_model(
            model,
            TEST_PROMPTS,
            timeout=args.timeout,
            warmup=not args.no_warmup,
        )
        all_results.append(result)

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Save to JSON ──────────────────────────────────────────────────────────
    output = {
        "benchmark_meta": {
            "run_at"    : datetime.now(timezone.utc).isoformat(),
            "models"    : models,
            "num_prompts": len(TEST_PROMPTS),
        },
        "results": all_results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✔ Results saved to: {args.out}")


if __name__ == "__main__":
    main()
