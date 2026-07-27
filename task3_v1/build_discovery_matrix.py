"""Build the 1,000 x 128 Task 3 pilot discovery matrix from WikiText.

The matrix contains normalized static J-lens coordinates at four layers for
32 token-anchored concepts. It is an observational feature matrix, not a
causal graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import jlens
import numpy as np
import torch
import transformers
from jlens.hooks import ActivationRecorder

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
CONCEPTS = [
    " water",
    " fire",
    " music",
    " danger",
    " Italy",
    " code",
    " animal",
    " happy",
    " money",
    " doctor",
    " city",
    " truth",
    " false",
    " love",
    " anger",
    " fear",
    " food",
    " sleep",
    " work",
    " school",
    " family",
    " war",
    " peace",
    " science",
    " art",
    " language",
    " number",
    " time",
    " future",
    " past",
    " safe",
    " risk",
]


def clean_wikitext(line: str) -> str:
    line = (
        line.replace("@-@", "-")
        .replace("@.@", ".")
        .replace("@,@", ",")
        .replace("<unk>", "unknown")
    )
    line = re.sub(r"\s+", " ", line).strip()
    return line


def select_grouped_prompts(
    corpus: Path,
    tokenizer,
    count: int,
    max_tokens: int,
) -> tuple[list[str], list[str], dict]:
    """Select paragraphs round-robin by top-level article."""
    grouped: dict[str, deque[str]] = defaultdict(deque)
    article = "unknown"
    raw_text = corpus.read_text(encoding="utf-8")
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        top_heading = re.fullmatch(r"= ([^=].*?) =", line)
        if top_heading:
            article = clean_wikitext(top_heading.group(1))
            continue
        if not line or line.startswith("="):
            continue
        cleaned = clean_wikitext(line)
        if len(cleaned.split()) < 20:
            continue
        token_ids = tokenizer.encode(cleaned, add_special_tokens=False)
        token_ids = token_ids[:max_tokens]
        prompt = tokenizer.decode(token_ids).strip()
        if prompt:
            grouped[article].append(prompt)

    prompts: list[str] = []
    groups: list[str] = []
    active = deque(sorted(grouped))
    while active and len(prompts) < count:
        group = active.popleft()
        prompts.append(grouped[group].popleft())
        groups.append(group)
        if grouped[group]:
            active.append(group)
    if len(prompts) != count:
        raise ValueError(
            f"Corpus yielded only {len(prompts)} eligible prompts; need {count}"
        )
    stats = {
        "corpus_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "eligible_paragraphs": int(sum(len(values) for values in grouped.values()))
        + count,
        "article_groups_available": len(grouped),
        "article_groups_selected": len(set(groups)),
        "selection": "deterministic round-robin by top-level article",
    }
    return prompts, groups, stats


def capture(model, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(ids)
        return {layer: recorder.activations[layer].detach() for layer in layers}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3") / "outputs" / "discovery",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.corpus.is_file():
        raise FileNotFoundError(args.corpus)

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    print("Loading cached model and Jacobian lens...", flush=True)
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

    token_ids = []
    for concept in CONCEPTS:
        ids = tokenizer.encode(concept, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Not one token: {concept!r} -> {ids}")
        token_ids.append(ids[0])

    prompts, group_ids, corpus_stats = select_grouped_prompts(
        args.corpus, tokenizer, args.count, args.max_tokens
    )
    prompt_token_counts = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    print(
        f"Selected {len(prompts)} prompts from "
        f"{len(set(group_ids))} article groups.",
        flush=True,
    )

    vectors_by_layer = {}
    for layer in layers:
        jacobian = lens.jacobians[layer].to(device="cuda", dtype=torch.float32)
        raw_vectors = jacobian.T @ model._lm_head.weight[token_ids].float().T
        vectors_by_layer[layer] = torch.nn.functional.normalize(
            raw_vectors, dim=0
        )

    rows = []
    prompt_seconds = []
    print("Projecting activations into 128 pilot coordinates...", flush=True)
    for prompt_index, prompt in enumerate(prompts):
        tick = time.time()
        acts = capture(model, prompt, layers)
        row = []
        for layer in layers:
            residual = acts[layer][0, -1].float()
            row.extend((residual @ vectors_by_layer[layer]).cpu().tolist())
        rows.append(row)
        torch.cuda.synchronize()
        prompt_seconds.append(time.time() - tick)
        if (prompt_index + 1) % 100 == 0:
            print(f"  projected {prompt_index + 1}/{len(prompts)}", flush=True)

    matrix = np.asarray(rows, dtype=np.float32)
    expected_shape = (args.count, len(layers) * len(CONCEPTS))
    if matrix.shape != expected_shape:
        raise RuntimeError(f"Expected matrix shape {expected_shape}, got {matrix.shape}")
    column_std = matrix.std(axis=0, ddof=1)
    columns = [
        {
            "index": index,
            "layer": layer,
            "concept": concept,
            "token_id": token_id,
        }
        for index, (layer, concept, token_id) in enumerate(
            (layer, concept, token_id)
            for layer in layers
            for concept, token_id in zip(CONCEPTS, token_ids)
        )
    ]
    record = {
        "status": "observational_pilot_matrix_not_causal_graph",
        "model": MODEL_NAME,
        "lens_revision": LENS_REVISION,
        "corpus": str(args.corpus),
        "corpus_stats": corpus_stats,
        "shape": list(matrix.shape),
        "layers": layers,
        "concept_count": len(CONCEPTS),
        "columns": columns,
        "prompt_token_count_min": int(np.min(prompt_token_counts)),
        "prompt_token_count_median": float(np.median(prompt_token_counts)),
        "prompt_token_count_max": int(np.max(prompt_token_counts)),
        "nan_count": int(np.isnan(matrix).sum()),
        "zero_variance_columns": int(np.sum(column_std < 1e-8)),
        "column_std_min": float(column_std.min()),
        "column_std_median": float(np.median(column_std)),
        "column_std_max": float(column_std.max()),
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
    np.save(args.output_dir / "discovery_matrix_1000x128.npy", matrix)
    np.savez_compressed(
        args.output_dir / "discovery_matrix_1000x128.npz",
        X=matrix,
        prompts=np.asarray(prompts),
        group_ids=np.asarray(group_ids),
        columns=np.asarray(
            [json.dumps(column, ensure_ascii=False) for column in columns]
        ),
    )
    (args.output_dir / "discovery_matrix_1000x128.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
