"""Collect token-anchored linear J-space coordinates from Qwen with J-lens.

Stages:

* ``smoke``: 50 clean prompts plus a few exact ``lens.apply`` readouts.
* ``discovery``: 500 clean/intervention prompt pairs (1000 CauScale rows).
* ``heldout``: 100 disjoint clean/intervention pairs for causal validation.

The measured coordinate is the residual projection onto the fixed direction
``normalize(J_l.T @ (gamma * W_U[token]))``.  Interventions use the minimum-
norm dual of the selected directions at a layer, so hard-setting one selected
coordinate does not directly change the other selected coordinates at that
same layer.  Exact J-lens logits are retained as a qualitative gate only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

import jlens
from jlens.hooks import ActivationRecorder


THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
JLENS_SOURCE = WORKSPACE_ROOT / "external" / "jacobian-lens"
JLENS_DATA = JLENS_SOURCE / "data"
DEFAULT_MODEL = WORKSPACE_ROOT / ".hf-jlens" / "model" / "Qwen3.5-4B"
DEFAULT_LENS = (
    WORKSPACE_ROOT
    / ".hf-jlens"
    / "lens"
    / "qwen3.5-4b"
    / "jlens"
    / "Salesforce-wikitext"
    / "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
sys.path.insert(0, str(THIS_FILE.parent))

from contracts import SCHEMA_VERSION, save_json, sha256_file, validate_dataset  # noqa: E402


def recursive_prompts(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"prompt", "stimulus"} and isinstance(child, str) and len(child.strip()) >= 20:
                yield child.strip()
            yield from recursive_prompts(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_prompts(child)


def prompt_corpus(seed: int) -> list[dict[str, str]]:
    records: list[tuple[str, str]] = []
    for path in sorted(JLENS_DATA.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = path.relative_to(JLENS_SOURCE).as_posix()
        records.extend((prompt, source) for prompt in recursive_prompts(payload))
        if isinstance(payload, dict) and "categories" in payload:
            for category in payload["categories"]:
                for argument in category.get("args", []):
                    for function in category.get("funcs", []):
                        template = function.get("template")
                        if template:
                            records.append((template.format(arg=argument).strip(), source))

    unique: dict[str, str] = {}
    for prompt, source in records:
        unique.setdefault(prompt, source)
    corpus = [
        {
            "prompt_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "prompt": prompt,
            "source": source,
        }
        for prompt, source in unique.items()
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(corpus))
    return [corpus[int(index)] for index in order]


def one_token_id(tokenizer, surface: str) -> tuple[int, str]:
    for text in (" " + surface, surface):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0]), text
    raise ValueError(f"configured concept {surface!r} is not one token with this tokenizer")


def selected_layers(n_layers: int, fractions: list[float]) -> list[int]:
    layers = [int(round(n_layers * fraction)) for fraction in fractions]
    if len(set(layers)) != len(layers) or any(layer < 0 or layer >= n_layers - 1 for layer in layers):
        raise ValueError(f"invalid layer fractions {fractions} for a {n_layers}-layer model: {layers}")
    return layers


def build_directions(model, lens, layers: list[int], token_ids: dict[str, int]):
    gamma = getattr(model._final_norm, "weight", None)
    directions = []
    nodes = []
    labels = {}
    column = 0
    for layer in layers:
        jacobian = lens.jacobians[layer].float().cpu()
        layer_vectors = []
        for concept, token_id in token_ids.items():
            unembed = model._lm_head.weight[token_id].detach().float().cpu()
            if gamma is not None:
                unembed = unembed * gamma.detach().float().cpu()
            direction = jacobian.T @ unembed
            direction = direction / direction.norm().clamp_min(1e-12)
            layer_vectors.append(direction)
            node_id = f"z{column:02d}"
            nodes.append(
                {
                    "node_id": node_id,
                    "column": column,
                    "kind": "jlens_token_direction_projection",
                    "layer": layer,
                    "concept": concept,
                    "token_id": token_id,
                    "position_rule": "last_prompt_token",
                }
            )
            labels[node_id] = f"{concept} representation at transformer layer {layer}"
            column += 1
        directions.append(torch.stack(layer_vectors))
    return torch.stack(directions), nodes, labels


def build_dual_directions(directions: torch.Tensor, layers: list[int]):
    """Return row-wise minimum-norm duals for each layer's measurement vectors.

    For a layer with measurement matrix ``V`` (one unit direction per row), the
    returned matrix is ``U = (V V^T)^-1 V``.  Consequently ``V U^T = I`` and an
    update ``delta * U[j]`` changes selected coordinate ``j`` by ``delta`` while
    leaving the other selected coordinates at that layer unchanged in exact
    arithmetic.
    """
    if directions.ndim != 3 or directions.shape[0] != len(layers):
        raise ValueError("directions must be [n_layers, n_coordinates_per_layer, d_model]")
    duals = []
    diagnostics = []
    for layer_index, layer in enumerate(layers):
        vectors = directions[layer_index].double()
        singular_values = torch.linalg.svdvals(vectors)
        rank = int(torch.linalg.matrix_rank(vectors).item())
        if rank != vectors.shape[0]:
            raise RuntimeError(
                f"selected directions at layer {layer} are rank deficient: "
                f"rank={rank}, expected={vectors.shape[0]}"
            )
        gram = vectors @ vectors.T
        dual = torch.linalg.solve(gram, vectors)
        coordinate_map = vectors @ dual.T
        identity_error = float(
            (coordinate_map - torch.eye(vectors.shape[0], dtype=vectors.dtype)).abs().max()
        )
        if identity_error > 1e-8:
            raise RuntimeError(
                f"dual construction failed at layer {layer}: max |V U^T - I|={identity_error:.3g}"
            )
        diagnostics.append(
            {
                "layer": int(layer),
                "rank": rank,
                "condition_number": float(singular_values.max() / singular_values.min()),
                "max_identity_error_float64": identity_error,
                "max_pairwise_measurement_cosine": float(
                    (gram - torch.diag_embed(torch.diagonal(gram))).abs().max()
                ),
            }
        )
        duals.append(dual.float())
    return torch.stack(duals), diagnostics


def hidden_from_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"unsupported transformer block output {type(output)!r}")


def replace_hidden(output, hidden):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise TypeError(f"unsupported transformer block output {type(output)!r}")


@contextlib.contextmanager
def hard_set_coordinate(
    model,
    *,
    layer: int,
    measurement_direction: torch.Tensor,
    dual_direction: torch.Tensor,
    target_value: float,
):
    """Set one measured coordinate to a fixed value at a block's output.

    The target is independent of the prompt-specific pre-intervention value.
    Computation is done in float32 and cast back to the model dtype; downstream
    QA measures the small error introduced by that cast.
    """
    def hook(_module, _inputs, output):
        hidden = hidden_from_output(output)
        edited = hidden.clone()
        measurement = measurement_direction.to(device=hidden.device, dtype=torch.float32)
        dual = dual_direction.to(device=hidden.device, dtype=torch.float32)
        current = hidden[:, -1, :].float() @ measurement
        delta = float(target_value) - current
        updated = hidden[:, -1, :].float() + delta[:, None] * dual
        edited[:, -1, :] = updated.to(dtype=hidden.dtype)
        return replace_hidden(output, edited)

    handle = model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def coordinates_for_prompt(
    model,
    prompt: str,
    layers: list[int],
    directions_gpu: dict[int, torch.Tensor],
    *,
    intervention: dict | None = None,
    max_length: int,
) -> np.ndarray:
    ids = model.encode(prompt, max_length=max_length)
    edit = (
        hard_set_coordinate(
            model,
            layer=int(intervention["layer"]),
            measurement_direction=intervention["measurement_direction"],
            dual_direction=intervention["dual_direction"],
            target_value=float(intervention["target_value"]),
        )
        if intervention is not None
        else contextlib.nullcontext()
    )
    # The edit hook is registered first; the recorder therefore observes the
    # post-intervention tensor at the target layer.
    with torch.inference_mode(), edit:
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(ids)
        rows = []
        for layer in layers:
            residual = recorder.activations[layer][0, -1].detach().float()
            rows.append((directions_gpu[layer] @ residual).cpu())
    return torch.cat(rows).numpy().astype(np.float32)


def qualitative_readouts(model, lens, tokenizer, prompts: list[dict], layers: list[int], max_length: int):
    records = []
    for prompt_record in prompts:
        logits, model_logits, _ = lens.apply(
            model,
            prompt_record["prompt"],
            layers=layers,
            positions=[-1],
            max_seq_len=max_length,
        )
        by_layer = {}
        for layer in layers:
            top = logits[layer][0].topk(5)
            by_layer[str(layer)] = [
                {"token_id": int(token_id), "token": tokenizer.decode([int(token_id)]), "logit": float(value)}
                for value, token_id in zip(top.values, top.indices)
            ]
        final_top = model_logits[0].topk(5)
        records.append(
            {
                "prompt_id": prompt_record["prompt_id"],
                "prompt": prompt_record["prompt"],
                "jlens_top5": by_layer,
                "model_top5": [
                    {"token_id": int(token_id), "token": tokenizer.decode([int(token_id)]), "logit": float(value)}
                    for value, token_id in zip(final_top.values, final_top.indices)
                ],
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def load_runtime(model_path: Path, lens_path: Path):
    if not model_path.is_dir():
        raise FileNotFoundError(f"local Qwen model is missing: {model_path}")
    if not lens_path.is_file():
        raise FileNotFoundError(f"local Jacobian lens is missing: {lens_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).cuda()
    model = jlens.from_hf(hf_model, tokenizer, compile=False)
    lens = jlens.JacobianLens.load(str(lens_path))
    if model.d_model != lens.d_model:
        raise RuntimeError(f"model/lens width mismatch: {model.d_model} != {lens.d_model}")
    return hf_model, tokenizer, model, lens


def save_dataset(
    output: Path,
    *,
    X: np.ndarray,
    interventions: np.ndarray,
    strengths: np.ndarray,
    directions: np.ndarray,
    dual_directions: np.ndarray,
    nodes: list[dict],
    labels: dict[str, str],
    row_metadata: list[dict],
    manifest: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "X.npy", X.astype(np.float32))
    np.save(output / "interventions.npy", interventions.astype(np.uint8))
    np.save(output / "intervention_strengths.npy", strengths.astype(np.float32))
    np.save(output / "directions.npy", directions.astype(np.float32))
    np.save(output / "dual_directions.npy", dual_directions.astype(np.float32))
    save_json(output / "nodes.json", nodes)
    save_json(output / "labels.json", labels)
    write_jsonl(output / "rows.jsonl", row_metadata)
    manifest["files"] = {
        name: sha256_file(output / name)
        for name in [
            "X.npy",
            "interventions.npy",
            "intervention_strengths.npy",
            "directions.npy",
            "dual_directions.npy",
            "nodes.json",
            "labels.json",
            "rows.jsonl",
        ]
    }
    save_json(output / "manifest.json", manifest)
    validate_dataset(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "discovery", "heldout"], default="smoke")
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "config" / "smoke.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualitative-prompts", type=int, default=3)
    parser.add_argument("--qa-target-tolerance-sigma", type=float, default=0.10)
    parser.add_argument("--qa-off-target-tolerance-sigma", type=float, default=0.10)
    args = parser.parse_args()
    if args.qa_target_tolerance_sigma <= 0 or args.qa_off_target_tolerance_sigma <= 0:
        raise ValueError("QA tolerances must be positive fractions of a calibration sigma")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    jconfig = config["jlens"]
    output = args.output or EXPERIMENT_ROOT / "runs" / f"jspace_{args.stage}"
    if output.is_dir() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing experiment output {output}; "
            "pass --output with a new directory"
        )
    corpus = prompt_corpus(int(config["seed"]))
    if len(corpus) < 600:
        raise RuntimeError(f"expected at least 600 unique official prompts; found {len(corpus)}")
    if args.stage == "smoke":
        prompts = corpus[: int(jconfig["readout_smoke_prompts"])]
    elif args.stage == "discovery":
        prompts = corpus[: int(jconfig["observational_rows"])]
    else:
        start = int(jconfig["observational_rows"])
        prompts = corpus[start : start + int(jconfig["heldout_prompt_pairs"])]

    started = time.time()
    hf_model, tokenizer, model, lens = load_runtime(args.model, args.lens)
    layers = selected_layers(model.n_layers, jconfig["layer_fractions"])
    if any(layer not in lens.source_layers for layer in layers):
        raise RuntimeError(f"selected layers {layers} not covered by lens {lens.source_layers}")
    token_info = {
        concept: one_token_id(tokenizer, concept) for concept in jconfig["concept_surfaces"]
    }
    token_ids = {concept: token_id for concept, (token_id, _) in token_info.items()}
    directions_cpu, nodes, labels = build_directions(model, lens, layers, token_ids)
    dual_directions_cpu, dual_diagnostics = build_dual_directions(directions_cpu, layers)
    directions_gpu = {
        layer: directions_cpu[layer_index].to(model.input_device)
        for layer_index, layer in enumerate(layers)
    }
    flattened_directions = directions_cpu.reshape(-1, model.d_model).numpy()
    flattened_dual_directions = dual_directions_cpu.reshape(-1, model.d_model).numpy()

    qualitative = qualitative_readouts(
        model,
        lens,
        tokenizer,
        prompts[: args.qualitative_prompts],
        layers,
        int(jconfig["max_sequence_length"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "qualitative_readouts.json", qualitative)

    clean_rows = []
    for index, prompt_record in enumerate(prompts):
        clean_rows.append(
            coordinates_for_prompt(
                model,
                prompt_record["prompt"],
                layers,
                directions_gpu,
                max_length=int(jconfig["max_sequence_length"]),
            )
        )
        if (index + 1) % 25 == 0 or index + 1 == len(prompts):
            print(f"clean {index + 1}/{len(prompts)}", flush=True)
    clean_X = np.stack(clean_rows).astype(np.float32)

    d = clean_X.shape[1]
    if d != len(nodes):
        raise AssertionError("coordinate/node count mismatch")
    if args.stage == "smoke":
        X = clean_X
        interventions = np.zeros_like(X, dtype=np.uint8)
        strengths = np.zeros_like(X, dtype=np.float32)
        rows = [
            {**record, "regime": "clean", "pair_id": record["prompt_id"]}
            for record in prompts
        ]
        evidence_status = "real_jlens_readout_smoke_not_causcale_evidence"
    else:
        calibration_n = min(400, len(clean_X))
        coordinate_mean = clean_X[:calibration_n].mean(axis=0)
        coordinate_std = clean_X[:calibration_n].std(axis=0)
        if np.any(coordinate_std < 1e-6):
            bad = np.flatnonzero(coordinate_std < 1e-6).tolist()
            raise RuntimeError(f"near-constant J-space coordinates: {bad}")
        rng = np.random.default_rng(int(config["seed"]) + (0 if args.stage == "discovery" else 1))
        targets = np.resize(np.arange(d), len(prompts))
        rng.shuffle(targets)
        levels = np.asarray(jconfig["intervention_strength_sigma"], dtype=float)
        if levels.ndim != 1 or len(levels) == 0 or np.any(levels <= 0):
            raise ValueError("intervention_strength_sigma must contain positive values")
        signed_conditions = np.asarray(
            [sign * level for level in levels for sign in (1.0, -1.0)],
            dtype=float,
        )
        # Balance sign and magnitude *within each intervention target*.  Pairing
        # sign with the loop-index parity would confound +/− with 1σ/2σ.
        signed_levels = np.empty(len(prompts), dtype=float)
        for target in range(d):
            target_rows = np.flatnonzero(targets == target)
            # Rotate the condition receiving the unavoidable extra replicate
            # (e.g. 5 or 25 rows per target) so it is balanced globally.
            target_conditions = np.roll(signed_conditions, -(target % len(signed_conditions)))
            assignments = np.resize(target_conditions, len(target_rows)).copy()
            rng.shuffle(assignments)
            signed_levels[target_rows] = assignments
        edited_rows = []
        interventions_edited = np.zeros((len(prompts), d), dtype=np.uint8)
        strengths_edited = np.zeros((len(prompts), d), dtype=np.float32)
        edited_metadata = []
        qa_records = []
        for index, (prompt_record, target) in enumerate(zip(prompts, targets)):
            target = int(target)
            signed_sigma = float(signed_levels[index])
            target_value = float(
                coordinate_mean[target] + signed_sigma * coordinate_std[target]
            )
            requested_coordinate_delta = target_value - float(clean_X[index, target])
            node = nodes[target]
            layer_index = layers.index(int(node["layer"]))
            concept_index = list(token_ids).index(node["concept"])
            measurement_direction = directions_cpu[layer_index, concept_index]
            dual_direction = dual_directions_cpu[layer_index, concept_index]
            edited_row = coordinates_for_prompt(
                model,
                prompt_record["prompt"],
                layers,
                directions_gpu,
                intervention={
                    "layer": node["layer"],
                    "measurement_direction": measurement_direction,
                    "dual_direction": dual_direction,
                    "target_value": target_value,
                },
                max_length=int(jconfig["max_sequence_length"]),
            )
            edited_rows.append(edited_row)
            same_layer = np.asarray(
                [i for i, candidate in enumerate(nodes) if candidate["layer"] == node["layer"]],
                dtype=np.int64,
            )
            same_layer_off_target = same_layer[same_layer != target]
            target_error_sigma = abs(float(edited_row[target]) - target_value) / float(
                coordinate_std[target]
            )
            off_target_delta_abs = np.abs(
                edited_row[same_layer_off_target] - clean_X[index, same_layer_off_target]
            )
            off_target_delta_sigma = (
                off_target_delta_abs / coordinate_std[same_layer_off_target]
            )
            max_off_target_delta_sigma = (
                float(off_target_delta_sigma.max()) if off_target_delta_sigma.size else 0.0
            )
            qa_pass = (
                target_error_sigma <= args.qa_target_tolerance_sigma
                and max_off_target_delta_sigma <= args.qa_off_target_tolerance_sigma
            )
            qa_record = {
                "prompt_id": prompt_record["prompt_id"],
                "target_node": node["node_id"],
                "target_value": target_value,
                "achieved_target_value": float(edited_row[target]),
                "target_error_abs": abs(float(edited_row[target]) - target_value),
                "target_error_sigma": target_error_sigma,
                "same_layer_off_target_delta_abs": {
                    nodes[int(candidate)]["node_id"]: float(delta)
                    for candidate, delta in zip(same_layer_off_target, off_target_delta_abs)
                },
                "same_layer_off_target_delta_sigma": {
                    nodes[int(candidate)]["node_id"]: float(delta)
                    for candidate, delta in zip(same_layer_off_target, off_target_delta_sigma)
                },
                "max_same_layer_off_target_delta_sigma": max_off_target_delta_sigma,
                "pass": qa_pass,
            }
            qa_records.append(qa_record)
            interventions_edited[index, target] = 1
            strengths_edited[index, target] = signed_sigma
            edited_metadata.append(
                {
                    **prompt_record,
                    "regime": "single_coordinate_injection",
                    "pair_id": prompt_record["prompt_id"],
                    "target_node": node["node_id"],
                    "signed_sigma": signed_sigma,
                    "configured_setpoint": target_value,
                    "configured_target_shift": requested_coordinate_delta,
                    "realized_target_shift": float(edited_row[target] - clean_X[index, target]),
                    "hard_do_target_value": target_value,
                    "requested_coordinate_delta": requested_coordinate_delta,
                    "residual_l2_amplitude": abs(requested_coordinate_delta)
                    * float(dual_direction.norm()),
                    "intervention_qa": qa_record,
                }
            )
            if (index + 1) % 25 == 0 or index + 1 == len(prompts):
                print(f"intervention {index + 1}/{len(prompts)}", flush=True)
        edited_X = np.stack(edited_rows).astype(np.float32)
        failed_qa = [record for record in qa_records if not record["pass"]]
        qa_summary = {
            "target_tolerance_sigma": float(args.qa_target_tolerance_sigma),
            "same_layer_off_target_tolerance_sigma": float(
                args.qa_off_target_tolerance_sigma
            ),
            "rows_checked": len(qa_records),
            "failed_rows": len(failed_qa),
            "max_target_error_sigma": max(
                record["target_error_sigma"] for record in qa_records
            ),
            "max_same_layer_off_target_delta_sigma": max(
                record["max_same_layer_off_target_delta_sigma"] for record in qa_records
            ),
        }
        if failed_qa:
            raise RuntimeError(
                "hard-intervention isolation QA failed; "
                f"summary={qa_summary}, first_failures={failed_qa[:5]}"
            )
        X = np.concatenate([clean_X, edited_X], axis=0)
        interventions = np.concatenate(
            [np.zeros_like(clean_X, dtype=np.uint8), interventions_edited], axis=0
        )
        strengths = np.concatenate(
            [np.zeros_like(clean_X, dtype=np.float32), strengths_edited], axis=0
        )
        rows = [
            {**record, "regime": "clean", "pair_id": record["prompt_id"]}
            for record in prompts
        ] + edited_metadata
        evidence_status = (
            "real_jspace_discovery_dataset"
            if args.stage == "discovery"
            else "real_jspace_heldout_intervention_dataset"
        )

    dev_n = min(400, len(clean_X))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": f"qwen3.5-4b_jlens_{args.stage}",
        "evidence_status": evidence_status,
        "seed": int(config["seed"]),
        "source_prompt_count": len(corpus),
        "base_prompt_count": len(prompts),
        "paired_design": args.stage != "smoke",
        "intervention_design": None
        if args.stage == "smoke"
        else {
            "mode": "hard_set_coordinate",
            "assignment": "balanced_signed_strengths_within_target",
            "signed_sigma_conditions": signed_conditions.astype(float).tolist(),
            "hard_do_definition": "coordinate_j := calibration_mean_j + signed_sigma * calibration_std_j",
            "intervention_strengths_semantics": (
                "signed_sigma of the configured setpoint relative to calibration mean/std; "
                "not an additive row-specific shift"
            ),
            "injection_vector": "same_layer_minimum_norm_dual",
            "isolation_scope": (
                "selected coordinates at the intervened layer only; downstream coordinates may "
                "change causally and unselected residual-space directions are unconstrained"
            ),
        },
        "intervention_qa": None if args.stage == "smoke" else qa_summary,
        "n_samples": int(X.shape[0]),
        "n_nodes": int(X.shape[1]),
        "dev_clean_rows": list(range(dev_n)),
        "standardization": {
            "fit_split": f"first_{dev_n}_clean_rows",
            "mean": clean_X[:dev_n].mean(axis=0).astype(float).tolist(),
            "std": clean_X[:dev_n].std(axis=0).astype(float).tolist(),
        },
        "model": {
            "name": jconfig["model"],
            "local_path": str(args.model.resolve()),
            "n_layers": model.n_layers,
            "d_model": model.d_model,
            "dtype": str(next(hf_model.parameters()).dtype),
        },
        "jlens": {
            "source_commit": jconfig["source_commit"],
            "lens_repository": jconfig["lens_repository"],
            "lens_revision": jconfig["lens_revision"],
            "lens_path": str(args.lens.resolve()),
            "lens_sha256": sha256_file(args.lens),
            "lens_n_prompts": lens.n_prompts,
            "source_layers": lens.source_layers,
        },
        "coordinates": {
            "definition": "unit_projection_on_J_transpose_gamma_unembed_direction",
            "dual_definition": "U = (V V^T)^-1 V independently at each selected layer",
            "dual_diagnostics": dual_diagnostics,
            "layers": layers,
            "position_rule": "last_prompt_token",
            "concept_tokens": {
                concept: {"token_id": token_id, "encoded_surface": token_info[concept][1]}
                for concept, token_id in token_ids.items()
            },
        },
        "elapsed_seconds": time.time() - started,
    }
    save_dataset(
        output,
        X=X,
        interventions=interventions,
        strengths=strengths,
        directions=flattened_directions,
        dual_directions=flattened_dual_directions,
        nodes=nodes,
        labels=labels,
        row_metadata=rows,
        manifest=manifest,
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "shape": list(X.shape),
                "clean_rows": int(np.sum(interventions.sum(axis=1) == 0)),
                "intervention_rows": int(np.sum(interventions.sum(axis=1) == 1)),
                "elapsed_seconds": manifest["elapsed_seconds"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
