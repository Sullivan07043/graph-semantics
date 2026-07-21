"""Pure-PyTorch, prediction-only CauScale adapter for one custom graph.

The adapter intentionally bypasses the legacy Lightning metric/evaluation path:
there is no fake ground-truth graph and all three pair probabilities are kept.
For an unpadded single graph, explicit lengths imply that no tensor element is
padding; legitimate zeros must remain visible to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
CAUSCALE_ROOT = WORKSPACE_ROOT / "external" / "CauScale"
sys.path.insert(0, str(THIS_FILE.parent))

_MODULES_PATH = CAUSCALE_ROOT / "src" / "model" / "modules.py"
_MODULES_SPEC = importlib.util.spec_from_file_location("causcale_prediction_modules", _MODULES_PATH)
if _MODULES_SPEC is None or _MODULES_SPEC.loader is None:
    raise ImportError(f"cannot load CauScale modules from {_MODULES_PATH}")
_MODULES = importlib.util.module_from_spec(_MODULES_SPEC)
_MODULES_SPEC.loader.exec_module(_MODULES)
TopLayer = _MODULES.TopLayer
TwoStreamEncoder = _MODULES.TwoStreamEncoder

from contracts import load_json, sha256_file, validate_dataset  # noqa: E402
from graph_postprocess import build_dag  # noqa: E402


CAUSCALE_COMMIT = "9d4766bbe5efd118f9ae696545956e89ca8e4e4d"


class UnpaddedTwoStreamEncoder(TwoStreamEncoder):
    """Exact upstream forward for a single, unpadded graph with explicit lengths.

    Upstream infers padding with ``data == 0`` and ``precision == 0``. That is
    invalid for sparse J-space coordinates. The custom adapter never pads its
    sole graph, so the correct masks are both ``None``.
    """

    def forward(self, batch):
        datas = batch["data"]
        interventions = batch["interv"]
        graphs = batch["feats"]
        if datas.ndim != 3 or datas.shape != interventions.shape:
            raise ValueError("data and intervention tensors must both be [B,m,n]")
        if datas.shape[0] != 1 or graphs.shape[0] != 1:
            raise ValueError("prediction-only smoke adapter supports exactly one unpadded graph")
        if int(batch["data_sample_lengths"][0]) != datas.shape[1]:
            raise ValueError("data_sample_lengths does not match the unpadded tensor")
        if int(batch["data_node_lengths"][0]) != datas.shape[2]:
            raise ValueError("data_node_lengths does not match the unpadded tensor")
        if int(batch["number_of_nodes"][0]) != graphs.shape[1] or graphs.shape[1] != graphs.shape[2]:
            raise ValueError("number_of_nodes does not match the graph feature tensor")

        padding_mask_data = None
        padding_mask_graph = None
        datas = torch.stack([datas, interventions], dim=-1)
        datas = self.embed_data(datas)
        graphs = self.embed_graph(graphs[..., None])
        datas = self.data_layer_norm_before(datas)
        datas = self.dropout_module(datas)
        graphs = self.graph_layer_norm_before(graphs)
        graphs = self.dropout_module(graphs)
        datas = datas.permute(1, 2, 0, 3)
        graphs = graphs.permute(1, 2, 0, 3)
        for layer_index, layer in enumerate(self.layers):
            datas, graphs = layer(
                datas,
                graphs,
                padding_mask_data=padding_mask_data,
                padding_mask_graph=padding_mask_graph,
            )
            if (
                not self.args.disable_reduction_unit
                and layer_index >= self.reduction_k
                and layer_index % self.reduction_k == 0
            ):
                datas, padding_mask_data = self.reduction_unit(datas, padding_mask_data)
        datas = self.data_layer_norm_after(datas)
        datas = datas.permute(2, 0, 1, 3)
        graphs = self.graph_layer_norm_after(graphs)
        return graphs.permute(2, 0, 1, 3)


class PredictionOnlyCauScale(torch.nn.Module):
    def __init__(self, args: SimpleNamespace):
        super().__init__()
        self.encoder = UnpaddedTwoStreamEncoder(args)
        self.top_layer = TopLayer(embed_dim=args.embed_dim * 2, output_dim=3)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        graph_features = self.encoder(batch)
        d = graph_features.shape[1]
        pair_index = torch.triu_indices(d, d, offset=1, device=graph_features.device).T
        forward = graph_features[0, pair_index[:, 0], pair_index[:, 1]]
        backward = graph_features[0, pair_index[:, 1], pair_index[:, 0]]
        logits = self.top_layer(torch.cat([forward, backward], dim=-1))
        return pair_index, torch.softmax(logits, dim=-1)


def model_args() -> SimpleNamespace:
    return SimpleNamespace(
        transformer_num_layers=10,
        embed_dim=128,
        ffn_embed_dim=512,
        n_heads=16,
        dropout=0.1,
        scale_graph_rows=False,
        scale_graph_cols=False,
        scale_data_cols=True,
        attn_shape="hnij",
        head_dim=-1,
        disable_reduction_unit=False,
    )


def checkpoint_state(path: Path, model: torch.nn.Module) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state_dict mapping")
    expected = model.state_dict()
    filtered = {key: value for key, value in state.items() if key in expected}
    missing = sorted(set(expected) - set(filtered))
    shape_errors = {
        key: (tuple(filtered[key].shape), tuple(expected[key].shape))
        for key in filtered
        if filtered[key].shape != expected[key].shape
    }
    if missing or shape_errors:
        raise RuntimeError(
            f"checkpoint is incompatible; missing={missing[:8]} "
            f"shape_errors={shape_errors}"
        )
    model.load_state_dict(filtered, strict=True)
    return {
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_keys": len(filtered),
        "ignored_checkpoint_keys": sorted(set(state) - set(filtered)),
    }


def preprocess(
    X: np.ndarray,
    interventions: np.ndarray,
    *,
    feature_sample_size: int,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if len(X) < feature_sample_size:
        raise ValueError(f"need at least {feature_sample_size} rows; got {len(X)}")
    rng = np.random.default_rng(seed)
    feature_rows = rng.choice(len(X), size=feature_sample_size, replace=False)
    covariance = np.cov(X[feature_rows].T)
    precision = np.linalg.pinv(covariance, rcond=1e-10)
    if not np.isfinite(precision).all():
        raise ValueError("precision feature contains NaN or infinity")

    step = max(1, len(X) // sample_size)
    selected = np.arange(0, len(X), step)[:sample_size]
    data = X[selected].astype(np.float32)
    regimes = interventions[selected].astype(np.float32)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    data = (data - mean) / np.where(std == 0, 1.0, std)
    provenance = {
        "feature_seed": seed,
        "feature_sample_size": feature_sample_size,
        "feature_rows_sha256": hashlib.sha256(feature_rows.astype(np.int64).tobytes()).hexdigest(),
        "sample_size": int(len(selected)),
        "subsample_step": step,
        "standardization_mean": mean.reshape(-1).astype(float).tolist(),
        "standardization_std": std.reshape(-1).astype(float).tolist(),
        "exact_zero_data_values_retained": int(np.sum(data == 0)),
        "exact_zero_precision_values_retained": int(np.sum(precision == 0)),
    }
    return data, regimes, precision.astype(np.float32), provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-sample-size", type=int, default=500)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--feature-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--architecture-smoke",
        action="store_true",
        help="use random weights to validate tensor plumbing; output is marked non-evidence",
    )
    args = parser.parse_args()
    if args.architecture_smoke == (args.checkpoint is not None):
        raise ValueError("provide exactly one of --checkpoint or --architecture-smoke")
    if not CAUSCALE_ROOT.is_dir():
        raise FileNotFoundError(f"pinned CauScale source is missing: {CAUSCALE_ROOT}")

    report = validate_dataset(args.dataset)
    X = np.load(args.dataset / "X.npy")
    interventions = np.load(args.dataset / "interventions.npy")
    nodes = load_json(args.dataset / "nodes.json")
    node_names = np.array([node["node_id"] for node in nodes])
    data, regimes, precision, preprocessing = preprocess(
        X,
        interventions,
        feature_sample_size=args.feature_sample_size,
        sample_size=args.sample_size,
        seed=args.feature_seed,
    )

    model = PredictionOnlyCauScale(model_args()).to(args.device).eval()
    if args.checkpoint is not None:
        checkpoint = checkpoint_state(args.checkpoint, model)
        mode = "official_synthetic_checkpoint"
    else:
        torch.manual_seed(0)
        checkpoint = {"checkpoint_sha256": None, "checkpoint_keys": 0}
        mode = "random_weights_architecture_smoke_not_evidence"

    batch = {
        "data": torch.from_numpy(data).unsqueeze(0).to(args.device),
        "interv": torch.from_numpy(regimes).unsqueeze(0).to(args.device),
        "feats": torch.from_numpy(precision).unsqueeze(0).to(args.device),
        "data_sample_lengths": torch.tensor([len(data)], device=args.device),
        "data_node_lengths": torch.tensor([data.shape[1]], device=args.device),
        "number_of_nodes": torch.tensor([data.shape[1]], device=args.device),
    }
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        pair_index_tensor, pair_probs_tensor = model(batch)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    pair_index = pair_index_tensor.cpu().numpy().astype(np.int64)
    pair_probs = pair_probs_tensor.cpu().numpy().astype(np.float32)
    adjacency, directed, no_edge, decisions = build_dag(
        pair_index,
        pair_probs,
        d=data.shape[1],
        threshold=args.threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pair_index=pair_index,
        pair_probs=pair_probs,
        directed_probs=directed,
        no_edge_probs=no_edge,
        adjacency=adjacency,
        node_names=node_names,
        threshold=np.array(args.threshold, dtype=np.float32),
    )
    metadata = {
        "mode": mode,
        "evidence_status": "not_evidence" if args.architecture_smoke else "zero_shot_causcale_prediction",
        "causcale_source_commit": CAUSCALE_COMMIT,
        "dataset": report,
        "checkpoint": checkpoint,
        "model_args": vars(model_args()),
        "preprocessing": preprocessing,
        "device": args.device,
        "forward_seconds": elapsed,
        "threshold": args.threshold,
        "candidate_edges": len(decisions),
        "kept_edges": int(adjacency.sum()),
        "cycle_edges_removed": sum(not decision["kept"] for decision in decisions),
        "decisions": decisions,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": mode,
                "pair_probs_shape": list(pair_probs.shape),
                "kept_edges": int(adjacency.sum()),
                "forward_seconds": elapsed,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
