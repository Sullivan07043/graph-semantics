"""Minimal Task 3 J-lens/logit-lens preflight.

This is a feasibility check, not a causal-discovery experiment.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jlens
import torch
import transformers


MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)

PROMPTS = [
    "Fact: The currency used in the country shaped like a boot is",
    "Complete the analogy: bird is to sky as fish is to",
    "The programmer found an infinite loop because the condition was always",
]


def top_tokens(tokenizer, logits: torch.Tensor, k: int = 5) -> list[str]:
    return [tokenizer.decode([int(token)]) for token in logits.topk(k).indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("task3") / "outputs" / "preflight" / "jlens_preflight.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this preflight")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    load_started = time.time()
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO,
        filename=LENS_FILE,
        revision=LENS_REVISION,
    )
    load_seconds = time.time() - load_started

    layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        model.n_layers // 4 * 3,
        model.n_layers - 2,
    ]

    results = []
    for prompt in PROMPTS:
        prompt_started = time.time()
        lens_logits, model_logits, _ = lens.apply(
            model,
            prompt,
            layers=layers,
            positions=[-2],
        )
        logit_logits, _, _ = lens.apply(
            model,
            prompt,
            layers=layers,
            positions=[-2],
            use_jacobian=False,
        )
        torch.cuda.synchronize()
        results.append(
            {
                "prompt": prompt,
                "seconds": time.time() - prompt_started,
                "model_top5": top_tokens(tokenizer, model_logits[0]),
                "layers": {
                    str(layer): {
                        "jlens_top5": top_tokens(tokenizer, lens_logits[layer][0]),
                        "logit_lens_top5": top_tokens(
                            tokenizer, logit_logits[layer][0]
                        ),
                    }
                    for layer in layers
                },
            }
        )

    record = {
        "status": "preflight_only_not_causal_evidence",
        "model": MODEL_NAME,
        "lens_repo": LENS_REPO,
        "lens_revision": LENS_REVISION,
        "lens_file": LENS_FILE,
        "layers": layers,
        "load_seconds": load_seconds,
        "total_seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Keep Windows GBK consoles safe; the saved artifact remains UTF-8.
    print(json.dumps(record, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
