"""One-node minimum-norm dual write audit for Task 3 preflight.

This validates local coordinate control only. It is not evidence for a causal graph.
"""
from __future__ import annotations

import json
import time
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
CONCEPTS = [" water", " fire", " music", " danger", " Italy", " code", " animal", " happy"]
CALIBRATION_PROMPTS = [
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
AUDIT_PROMPT = "After hours in the sun, the exhausted hiker opened the container."


def capture(model, prompt: str, at: list[int]):
    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=at) as recorder:
        model.forward(ids)
        acts = {layer: recorder.activations[layer].detach() for layer in at}
    return ids, acts


def patched_capture(model, prompt: str, layer: int, final_layer: int, delta: torch.Tensor):
    def patch_hook(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        changed = tensor.clone()
        changed[:, -1, :] += delta.to(device=tensor.device, dtype=tensor.dtype)
        if torch.is_tensor(output):
            return changed
        return (changed, *output[1:])

    handle = model.layers[layer].register_forward_hook(patch_hook)
    try:
        return capture(model, prompt, [layer, final_layer])
    finally:
        handle.remove()


def top_changes(tokenizer, delta_logits: torch.Tensor, k: int = 8):
    values, ids = delta_logits.topk(k)
    return [
        {"token": tokenizer.decode([int(token)]), "delta_logit": float(value)}
        for value, token in zip(values, ids)
    ]


@torch.no_grad()
def main() -> None:
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

    layer = model.n_layers // 4 * 3
    final_layer = model.n_layers - 1
    token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in CONCEPTS]
    if any(len(tokenizer.encode(c, add_special_tokens=False)) != 1 for c in CONCEPTS):
        raise ValueError("All concepts must be single tokens")

    J = lens.jacobians[layer].to(device="cuda", dtype=torch.float32)
    unembed = model._lm_head.weight[token_ids].float()
    vectors = J.T @ unembed.T
    vectors = torch.nn.functional.normalize(vectors, dim=0)
    gram = vectors.T @ vectors
    ridge = 1e-4
    dual = vectors @ torch.linalg.inv(
        gram + ridge * torch.eye(len(CONCEPTS), device="cuda")
    )

    calibration = []
    for prompt in CALIBRATION_PROMPTS:
        _, acts = capture(model, prompt, [layer])
        h = acts[layer][0, -1].float()
        calibration.append((h @ vectors).cpu().numpy())
    calibration = np.asarray(calibration)
    coordinate_std = calibration.std(axis=0, ddof=1)

    _, control_acts = capture(model, AUDIT_PROMPT, [layer, final_layer])
    control_h = control_acts[layer][0, -1].float()
    control_coordinates = control_h @ vectors
    control_logits = model.unembed(
        control_acts[final_layer][0, -1].float()[None]
    )[0].float().cpu()

    source_index = CONCEPTS.index(" water")
    source_std = float(coordinate_std[source_index])
    interventions = []
    for dose in (-1.0, 1.0):
        requested = dose * source_std
        delta = dual[:, source_index] * requested
        _, changed_acts = patched_capture(
            model, AUDIT_PROMPT, layer, final_layer, delta
        )
        changed_h = changed_acts[layer][0, -1].float()
        changed_coordinates = changed_h @ vectors
        coordinate_change = (changed_coordinates - control_coordinates).cpu().numpy()
        standardized = coordinate_change / np.maximum(coordinate_std, 1e-8)
        changed_logits = model.unembed(
            changed_acts[final_layer][0, -1].float()[None]
        )[0].float().cpu()
        delta_logits = changed_logits - control_logits
        target_change = float(coordinate_change[source_index])
        interventions.append(
            {
                "dose_sd": dose,
                "requested_coordinate_change": requested,
                "achieved_coordinate_change": target_change,
                "target_error_sd": abs(target_change - requested) / source_std,
                "mean_abs_offtarget_sd": float(
                    np.mean(np.abs(np.delete(standardized, source_index)))
                ),
                "max_abs_offtarget_sd": float(
                    np.max(np.abs(np.delete(standardized, source_index)))
                ),
                "delta_residual_norm": float(delta.norm().item()),
                "source_output_token_delta_logit": float(
                    delta_logits[token_ids[source_index]].item()
                ),
                "top_positive_output_changes": top_changes(tokenizer, delta_logits),
            }
        )

    record = {
        "status": "local_write_audit_not_causal_evidence",
        "model": MODEL_NAME,
        "lens_revision": LENS_REVISION,
        "prompt": AUDIT_PROMPT,
        "layer": layer,
        "source_concept": CONCEPTS[source_index],
        "concepts": CONCEPTS,
        "coordinate_definition": "normalized static J-lens vectors; ridge dual write",
        "calibration_prompts": len(CALIBRATION_PROMPTS),
        "coordinate_std": coordinate_std.tolist(),
        "gram_condition_number": float(torch.linalg.cond(gram).item()),
        "ridge": ridge,
        "interventions": interventions,
        "total_seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    output = (
        Path("task3")
        / "outputs"
        / "preflight"
        / "jlens_dual_write_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
