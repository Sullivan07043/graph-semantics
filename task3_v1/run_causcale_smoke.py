"""Run the official CauScale checkpoint on the 128-node Task 3 pilot matrix.

This is an architecture and zero-shot OOD smoke test. With no ground-truth
graph or held-out intervention validation, its output is only a candidate
dependency score matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_args() -> SimpleNamespace:
    """Mirror the released config/inference.yaml plus parser defaults."""
    return SimpleNamespace(
        transformer_num_layers=10,
        embed_dim=128,
        n_heads=16,
        ffn_embed_dim=512,
        dropout=0.1,
        disable_reduction_unit=False,
        scale_data_cols=True,
        scale_graph_cols=False,
        scale_graph_rows=False,
        attn_shape="hnij",
        head_dim=-1,
        weight_decay=1e-6,
        lr=1e-4,
        use_self_defined_mask=True,
    )


def directed_probabilities(model, encoded: torch.Tensor, n: int):
    """Apply CauScale's released three-class head to every unordered pair."""
    output = encoded.permute(0, 3, 1, 2)
    forward = torch.triu(output, diagonal=1).permute(0, 2, 3, 1)
    backward = (
        torch.triu(output.transpose(2, 3), diagonal=1).permute(0, 2, 3, 1)
    )
    mask = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=output.device), diagonal=1
    )
    logits = model.top_layer(
        torch.cat([forward[0][mask], backward[0][mask]], dim=-1)
    )
    classes = torch.softmax(logits, dim=-1)
    upper_i, upper_j = torch.triu_indices(n, n, offset=1, device=output.device)
    directed = torch.zeros(n, n, device=output.device)
    no_edge = torch.zeros(n, n, device=output.device)
    directed[upper_i, upper_j] = classes[:, 1]
    directed[upper_j, upper_i] = classes[:, 2]
    no_edge[upper_i, upper_j] = classes[:, 0]
    no_edge[upper_j, upper_i] = classes[:, 0]
    return directed, no_edge


def arm_summary(
    name: str,
    directed: np.ndarray,
    no_edge: np.ndarray,
    allowed: np.ndarray,
    columns: list[dict],
) -> dict:
    allowed_scores = directed[allowed]
    offdiag = ~np.eye(len(directed), dtype=bool)
    top_indices = np.argwhere(allowed)
    top_order = np.argsort(allowed_scores)[::-1][:100]
    top_edges = []
    for rank, order_index in enumerate(top_order, start=1):
        source, target = top_indices[order_index]
        top_edges.append(
            {
                "rank": rank,
                "source_index": int(source),
                "target_index": int(target),
                "source_layer": int(columns[source]["layer"]),
                "target_layer": int(columns[target]["layer"]),
                "source_concept": columns[source]["concept"],
                "target_concept": columns[target]["concept"],
                "probability": float(directed[source, target]),
                "no_edge_probability": float(no_edge[source, target]),
            }
        )
    return {
        "name": name,
        "directed_probability_min": float(directed[offdiag].min()),
        "directed_probability_median": float(np.median(directed[offdiag])),
        "directed_probability_max": float(directed[offdiag].max()),
        "allowed_edge_count": int(allowed.sum()),
        "allowed_probability_median": float(np.median(allowed_scores)),
        "allowed_probability_p95": float(np.percentile(allowed_scores, 95)),
        "allowed_probability_max": float(allowed_scores.max()),
        "allowed_edges_ge_0_5": int(np.sum(allowed_scores >= 0.5)),
        "allowed_edges_ge_0_8": int(np.sum(allowed_scores >= 0.8)),
        "top_edges": top_edges,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--causcale-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3") / "outputs" / "causcale",
    )
    args = parser.parse_args()
    for path in [args.matrix, args.metadata, args.checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.causcale_src.is_dir():
        raise FileNotFoundError(args.causcale_src)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    sys.path.insert(0, str(args.causcale_src))
    from model import CauScale

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    matrix = np.load(args.matrix).astype(np.float32)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    columns = metadata["columns"]
    if matrix.shape != (1000, 128) or len(columns) != 128:
        raise ValueError(
            f"Expected matrix (1000, 128) and 128 columns; got "
            f"{matrix.shape} and {len(columns)}"
        )

    # Follow the released inference dataset: a 500-row precision prior is
    # computed before full-matrix column standardization.
    rng = np.random.RandomState(42)
    prior_rows = rng.choice(len(matrix), size=500, replace=False)
    precision = LedoitWolf().fit(matrix[prior_rows]).get_precision().astype(np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    standardized = (matrix - mean) / np.where(std == 0, 1.0, std)

    args_model = model_args()
    model = CauScale(args_model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model = model.eval().cuda()

    data = torch.from_numpy(standardized).unsqueeze(0).cuda()
    interventions = torch.zeros_like(data)
    layer_ids = np.asarray([column["layer"] for column in columns])
    allowed = layer_ids[:, None] < layer_ids[None, :]
    outputs = {}
    summaries = {}
    inference_seconds = {}
    priors = {
        "invcov": precision,
        "zero_prior": np.full_like(precision, 0.2),
    }
    print("Running official CauScale checkpoint on 128 nodes...", flush=True)
    for name, prior in priors.items():
        batch = {
            "data": data,
            "interv": interventions,
            "feats": torch.from_numpy(prior).unsqueeze(0).cuda(),
        }
        tick = time.time()
        encoded = model.encoder(batch)
        directed, no_edge = directed_probabilities(model, encoded, matrix.shape[1])
        torch.cuda.synchronize()
        inference_seconds[name] = time.time() - tick
        directed_np = directed.cpu().numpy()
        no_edge_np = no_edge.cpu().numpy()
        constrained = directed_np * allowed
        outputs[name] = {
            "directed": directed_np,
            "no_edge": no_edge_np,
            "constrained": constrained,
        }
        summaries[name] = arm_summary(
            name, directed_np, no_edge_np, allowed, columns
        )
        print(f"  completed {name} in {inference_seconds[name]:.3f}s", flush=True)

    allowed_invcov = outputs["invcov"]["directed"][allowed]
    allowed_zero = outputs["zero_prior"]["directed"][allowed]
    prior_sensitivity = {
        "allowed_edge_spearman": float(
            spearmanr(allowed_invcov, allowed_zero).statistic
        ),
        "mean_absolute_probability_change": float(
            np.mean(np.abs(allowed_invcov - allowed_zero))
        ),
        "top100_jaccard": float(
            len(
                set(np.argsort(allowed_invcov)[-100:])
                & set(np.argsort(allowed_zero)[-100:])
            )
            / len(
                set(np.argsort(allowed_invcov)[-100:])
                | set(np.argsort(allowed_zero)[-100:])
            )
        ),
    }
    record = {
        "status": "zero_shot_ood_smoke_not_causal_graph",
        "source_repository": "https://github.com/OpenCausaLab/CauScale",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_lightning_version": checkpoint.get(
            "pytorch-lightning_version"
        ),
        "state_dict_keys": len(checkpoint["state_dict"]),
        "strict_state_dict_load": True,
        "matrix": str(args.matrix),
        "matrix_sha256": sha256(args.matrix),
        "matrix_shape": list(matrix.shape),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "layer_constraint": "only earlier-layer -> later-layer scores retained",
        "allowed_directed_edges": int(allowed.sum()),
        "arms": summaries,
        "prior_sensitivity": prior_sensitivity,
        "inference_seconds": inference_seconds,
        "total_seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "interpretation": (
            "The checkpoint was trained on simulated SCMs; LLM coordinates are "
            "out of distribution. Scores require bootstrap stability and "
            "held-out activation intervention validation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "causcale_smoke_128.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "causcale_smoke_128.npz",
        invcov_directed=outputs["invcov"]["directed"],
        invcov_no_edge=outputs["invcov"]["no_edge"],
        invcov_constrained=outputs["invcov"]["constrained"],
        zero_directed=outputs["zero_prior"]["directed"],
        zero_no_edge=outputs["zero_prior"]["no_edge"],
        zero_constrained=outputs["zero_prior"]["constrained"],
        allowed_mask=allowed,
    )
    print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
