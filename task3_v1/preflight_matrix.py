"""Build a tiny J-lens-score matrix for Task 3 feasibility reporting.

The columns are selected-token J-lens logits, not validated causal variables
or the final sparse J-space coordinates.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jlens
import numpy as np
import torch
import transformers

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
CONCEPTS = [" water", " fire", " music", " danger", " Italy", " code", " animal", " happy"]
PROMPTS = [
    "A sailor watched dark clouds gather above the open sea.",
    "The engineer reviewed the failing test before changing the program.",
    "A traveler studied a map before choosing the next destination.",
    "The child listened carefully as the orchestra began to play.",
    "Smoke rose from the forest and the hikers moved away quickly.",
    "The chef filled a glass after the long afternoon in the kitchen.",
    "A biologist recorded how the creature moved through its habitat.",
    "The student smiled after receiving unexpectedly good news.",
    "The mechanic inspected the engine after hearing a strange sound.",
    "The doctor compared the scan with the earlier examination.",
    "A storm warning caused the harbor to close before sunset.",
    "The audience became quiet when the first notes began.",
    "The climber checked every rope before leaving the ground.",
    "A researcher repeated the calculation to find the mistake.",
    "The family planned a journey through several European cities.",
    "The rescue team noticed heat and thick smoke ahead.",
    "The farmer observed the behavior of a newly born creature.",
    "The developer traced the unexpected output to one condition.",
    "After the ceremony, everyone appeared cheerful and relaxed.",
    "The empty bottle was refilled before the group continued walking.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3") / "outputs" / "preflight",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        model.n_layers // 4 * 3,
        model.n_layers - 2,
    ]

    token_ids = {}
    for concept in CONCEPTS:
        ids = tokenizer.encode(concept, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Not one token: {concept!r} -> {ids}")
        token_ids[concept] = ids[0]

    rows, prompt_seconds = [], []
    for prompt in PROMPTS:
        tick = time.time()
        lens_logits, _, _ = lens.apply(model, prompt, layers=layers, positions=[-1])
        values = []
        for layer in layers:
            logits = lens_logits[layer][0]
            values.extend(float(logits[token_ids[c]].item()) for c in CONCEPTS)
        rows.append(values)
        torch.cuda.synchronize()
        prompt_seconds.append(time.time() - tick)

    matrix = np.asarray(rows, dtype=np.float32)
    columns = [
        {"layer": layer, "concept": concept, "token_id": token_ids[concept]}
        for layer in layers
        for concept in CONCEPTS
    ]
    std = matrix.std(axis=0)
    record = {
        "status": "preflight_jlens_scores_not_causal_variables",
        "model": MODEL_NAME,
        "lens_revision": LENS_REVISION,
        "shape": list(matrix.shape),
        "layers": layers,
        "concepts": columns,
        "nan_count": int(np.isnan(matrix).sum()),
        "zero_variance_columns": int((std < 1e-8).sum()),
        "column_std_min": float(std.min()),
        "column_std_median": float(np.median(std)),
        "column_std_max": float(std.max()),
        "mean_prompt_seconds": float(np.mean(prompt_seconds)),
        "total_seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "jlens_score_matrix_20x32.npz",
        X=matrix,
        prompts=np.asarray(PROMPTS),
        columns=np.asarray([json.dumps(c, ensure_ascii=False) for c in columns]),
    )
    (args.output_dir / "jlens_score_matrix_20x32.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
