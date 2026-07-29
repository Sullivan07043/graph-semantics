"""Audit-only decomposition of the frozen Stage-3 unrolled solver.

This module intentionally lives outside :mod:`v5`.  It does not change the
formal solver or any checkpoint.  Instead, it exposes the frozen objective as
six named components and replays the same functional-Adam updates with an
explicit term mask.  The canonical full-objective path delegates its loss
evaluation to ``v5.l2_solver.step_loss`` on every optimization step so that it
can be checked directly against ``solve_unrolled``.

The public entry points are:

``make_common_initial_state``
    Compute the oracle ALS initialization once and create a node-keyed
    residual initialization shared by every audit arm.

``component_losses``
    Evaluate the six frozen, WeightNet-weighted objective components.

``solve_audit``
    Run the audit-local functional Adam from a supplied common initialization
    and return embeddings, residuals, loss/gradient diagnostics, and numerical
    stability summaries.

No function in this file trains or mutates the encoder, LoRA, WeightNet, or
negation operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


TERM_NAMES: tuple[str, ...] = (
    "generation",
    "residual_norm",
    "residual_alignment",
    "independence",
    "bridge",
    "norm",
)

FULL_MASK: tuple[int, ...] = (1, 1, 1, 1, 1, 1)

TERM_MASKS: dict[str, tuple[int, ...] | None] = {
    "full_oracle": FULL_MASK,
    "generation_only_oracle": (1, 0, 0, 0, 0, 0),
    "oracle_without_generation": (0, 1, 1, 1, 1, 1),
    "residual_only_oracle": (0, 1, 1, 0, 0, 0),
    # The E0-double-prime brief explicitly groups the correlation bridge with
    # the independence diagnostic.
    "independence_only_oracle": (0, 0, 0, 1, 1, 0),
    "symmetrized_oracle": FULL_MASK,
    "markov_blanket_oracle": FULL_MASK,
    "same_module_graph": FULL_MASK,
    "reversed_full": FULL_MASK,
    "shuffled_full": FULL_MASK,
    # Closed-form baselines do not invoke the Stage-3 objective.
    "raw_correlation": None,
    "uniform": None,
}


class AuditSolverError(RuntimeError):
    """Raised when the audit solver cannot preserve the frozen interface."""


class NonFiniteOptimizationError(AuditSolverError):
    """Raised when a formal audit optimization produces a non-finite value."""


@dataclass(frozen=True)
class CommonInitialState:
    """Oracle initialization injected into every optimization arm.

    ``free_embeddings`` contains the exact NumPy ALS result for every free
    node.  ``residuals`` is keyed by every node in the reference graph:
    original generated nodes receive the exact draws used by the frozen
    solver, in its original order; remaining nodes receive subsequent draws
    from the same RNG.  Consequently, canonical full-oracle residuals are
    unchanged while nodes that become generated in a transformed arm still
    receive deterministic, node-stable initial values.
    """

    free_nodes: tuple[str, ...]
    free_embeddings: Mapping[str, np.ndarray]
    residuals: Mapping[str, np.ndarray]
    reference_gen_nodes: tuple[str, ...]
    node_order: tuple[str, ...]
    d: int
    seed: int


@dataclass
class AuditSolveResult:
    """Outputs from :func:`solve_audit`.

    The diagnostic rows are deliberately flat dictionaries so a caller can
    append provenance columns and write them directly to CSV.
    """

    embeddings: dict[str, np.ndarray]
    initial_embeddings: dict[str, np.ndarray]
    final_embeddings: dict[str, np.ndarray]
    initial_residuals: dict[str, np.ndarray]
    final_residuals: dict[str, np.ndarray]
    loss_terms: list[dict[str, Any]]
    gradient_norms: list[dict[str, Any]]
    trace: dict[str, Any]
    term_mask: dict[str, bool]
    canonical_full_path: bool

    def displacement_norm(self, node: str) -> float:
        """Return the raw L2 displacement of one free embedding."""

        return float(
            np.linalg.norm(
                np.asarray(self.final_embeddings[node], dtype=np.float64)
                - np.asarray(self.initial_embeddings[node], dtype=np.float64)
            )
        )


def normalize_term_mask(
    mask: str | Mapping[str, Any] | Sequence[Any],
) -> tuple[int, ...]:
    """Resolve an arm name, named mapping, or six-value sequence.

    Baseline names deliberately raise: they do not have a Stage-3 objective.
    """

    if isinstance(mask, str):
        if mask not in TERM_MASKS:
            raise AuditSolverError(f"unknown audit arm {mask!r}")
        resolved = TERM_MASKS[mask]
        if resolved is None:
            raise AuditSolverError(f"{mask!r} is a non-optimization baseline")
        return resolved
    if isinstance(mask, Mapping):
        unknown = set(mask) - set(TERM_NAMES)
        missing = set(TERM_NAMES) - set(mask)
        if unknown or missing:
            raise AuditSolverError(
                f"term mask keys differ from frozen terms; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        values = tuple(int(bool(mask[name])) for name in TERM_NAMES)
    else:
        if isinstance(mask, (str, bytes)) or len(mask) != len(TERM_NAMES):
            raise AuditSolverError(f"term mask must contain {len(TERM_NAMES)} values")
        values = tuple(int(bool(value)) for value in mask)
    if not any(values):
        raise AuditSolverError("at least one objective component must be active")
    return values


def term_mask_dict(mask: str | Mapping[str, Any] | Sequence[Any]) -> dict[str, bool]:
    """Return a named Boolean representation of a validated term mask."""

    values = normalize_term_mask(mask)
    return {name: bool(value) for name, value in zip(TERM_NAMES, values)}


def make_common_initial_state(
    frozen_solver: Any,
    reference_graph: Any,
    reference_weights: Mapping[tuple[str, str], float],
    labeled_embeddings: Mapping[str, np.ndarray],
    d: int,
    *,
    seed: int,
    residual_scale: float = 1e-3,
) -> CommonInitialState:
    """Build the common E0-double-prime initialization.

    The ALS call is the frozen solver's private stage-1 helper, not a
    reimplementation.  This is intentional: the full-oracle arm must begin
    at exactly the same point as ``solve_unrolled``.
    """

    if not isinstance(d, int) or d <= 0:
        raise AuditSolverError("d must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise AuditSolverError("seed must be an integer")
    if not np.isfinite(float(residual_scale)) or float(residual_scale) <= 0:
        raise AuditSolverError("residual_scale must be positive and finite")

    free, _anchors, initial = frozen_solver._stage1(
        reference_graph, dict(reference_weights), dict(labeled_embeddings), d
    )
    node_order = tuple(reference_graph.nodes)
    gen_nodes = tuple(node for node in node_order if reference_graph.parents(node))
    rng = np.random.default_rng(seed)
    residuals: dict[str, np.ndarray] = {}
    for node in gen_nodes:
        residuals[node] = rng.normal(0.0, float(residual_scale), d).astype(np.float64)
    for node in node_order:
        if node not in residuals:
            residuals[node] = rng.normal(0.0, float(residual_scale), d).astype(np.float64)

    free_embeddings = {
        node: np.asarray(initial[node], dtype=np.float64).copy() for node in free
    }
    if not all(np.isfinite(value).all() for value in free_embeddings.values()):
        raise NonFiniteOptimizationError("oracle ALS initialization is non-finite")
    return CommonInitialState(
        free_nodes=tuple(free),
        free_embeddings=free_embeddings,
        residuals=residuals,
        reference_gen_nodes=gen_nodes,
        node_order=node_order,
        d=d,
        seed=seed,
    )


def _zero_scalar(ctx: Mapping[str, Any]):
    """Create a scalar zero on the objective device without fake dependencies."""

    import torch

    return torch.zeros((), dtype=torch.float32, device=ctx["device"])


def component_losses(
    ctx: Mapping[str, Any],
    emb: Callable[[str], Any],
    wt: Callable[[tuple[str, str]], Any],
    free_embeddings: Mapping[str, Any],
    residual_vectors: Mapping[str, Any] | None,
    lam_zero: float,
    lam_norm: float,
    *,
    nw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the six frozen objective components.

    Operations, reductions, and WeightNet multipliers mirror
    ``v5.l2_solver.step_loss``.  In particular, the generation component keeps
    the residual channel whenever the frozen context enables it.  Therefore
    generation-only has unpenalized residual variables, and residual-only has
    no embedding dependence; these are properties of the frozen mathematics,
    not special cases introduced by the audit.
    """

    import torch

    W = ctx["W"]
    gen_nodes = ctx["gen_nodes"]
    anchors = ctx["At"]
    use_residual = ctx["use_res"] and residual_vectors is not None
    zero = _zero_scalar(ctx)

    negative_cache: dict[str, Any] = {}
    if ctx["neg_parents"]:
        transformed = ctx["neg_op"](
            torch.stack([emb(parent) for parent in ctx["neg_parents"]])
        )
        negative_cache = {
            parent: transformed[index]
            for index, parent in enumerate(ctx["neg_parents"])
        }

    generation = zero
    for index, node in enumerate(gen_nodes):
        total = None
        for parent in ctx["parents"][node]:
            edge = (parent, node)
            edge_weight = wt(edge)
            if parent in negative_cache and float(W.get(edge, 0.0)) < 0:
                contribution = torch.abs(edge_weight) * negative_cache[parent]
            else:
                contribution = edge_weight * emb(parent)
            total = contribution if total is None else total + contribution
        if use_residual:
            total = total + residual_vectors[node]
        target = anchors[node] if node in anchors else free_embeddings[node]
        value = ((target - total) ** 2).sum()
        generation = generation + (
            value if nw is None else nw["gen"][index] * value
        )

    residual_norm = zero
    residual_alignment = zero
    if use_residual:
        residual_matrix = torch.stack([residual_vectors[node] for node in gen_nodes])
        squared_norm = (residual_matrix**2).sum(1)
        residual_norm = ctx["residual"] * (
            squared_norm.mean()
            if nw is None
            else (nw["resnorm"] * squared_norm).mean()
        )
        if len(ctx["pc_nodes"]) > 1:
            aligned = torch.stack(
                [residual_vectors[node] for node in ctx["pc_nodes"]]
            )
            aligned = torch.nn.functional.normalize(aligned, dim=1)
            alignment_error = ((aligned @ aligned.T) - ctx["Pt"]) ** 2
            if nw is None:
                residual_alignment = (
                    ctx["lam_res"] * alignment_error[ctx["offdiag"]].mean()
                )
            else:
                anchor_weight = nw["anchor"]
                pair_weight = 0.5 * (
                    anchor_weight[:, None] + anchor_weight[None, :]
                )
                residual_alignment = (
                    ctx["lam_res"]
                    * (pair_weight * alignment_error)[ctx["offdiag"]].mean()
                )

    need_matrix = (
        (len(ctx["zp_pairs"]) and lam_zero > 0)
        or ctx["br_terms"] is not None
    )
    normalized = None
    pair_weight = None
    if need_matrix:
        matrix = torch.stack([emb(node) for node in ctx["all_nodes"]])
        normalized = torch.nn.functional.normalize(matrix, dim=1)
        pair_weight = None if nw is None else nw["node"]

    independence = zero
    if len(ctx["zp_pairs"]) and lam_zero > 0:
        value = (
            (normalized[ctx["ia"]] * normalized[ctx["ib"]]).sum(1)
        ) ** 2
        if pair_weight is not None:
            value = (
                0.5
                * (pair_weight[ctx["ia"]] + pair_weight[ctx["ib"]])
                * value
            )
        independence = lam_zero * value.mean()

    bridge = zero
    if ctx["br_terms"] is not None:
        left, right, floor, lam_upper = ctx["br_terms"]
        cosine = (
            normalized[left] * normalized[right]
        ).sum(1).abs()
        value = torch.relu(floor - cosine) ** 2
        if pair_weight is not None:
            value = (
                0.5 * (pair_weight[left] + pair_weight[right]) * value
            )
        bridge = lam_upper * value.mean()

    norm = zero
    if lam_norm > 0:
        norms = torch.stack(
            [free_embeddings[node].norm() for node in ctx["free"]]
        )
        value = (norms - 1.0) ** 2
        if nw is not None:
            value = nw["norm"] * value
        norm = lam_norm * value.mean()

    return {
        "generation": generation,
        "residual_norm": residual_norm,
        "residual_alignment": residual_alignment,
        "independence": independence,
        "bridge": bridge,
        "norm": norm,
    }


def _active_loss(
    components: Mapping[str, Any],
    mask_values: Sequence[int],
    parameters: Sequence[Any],
):
    """Sum active components, attaching a zero only for an empty graph loss."""

    loss: Any = 0.0
    for active, name in zip(mask_values, TERM_NAMES):
        if active:
            loss = loss + components[name]
    if not hasattr(loss, "requires_grad") or not loss.requires_grad:
        # A graph can make an otherwise active pair term empty.  Preserve a
        # valid zero-gradient optimization instead of failing autograd.
        loss = sum((parameter.sum() * 0.0 for parameter in parameters), loss)
    return loss


def _tensor_state(
    values: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    torch: Any,
    device: str,
    requires_grad: bool,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in names:
        tensor = torch.tensor(
            np.asarray(values[name], dtype=np.float64),
            dtype=torch.float32,
            device=device,
        )
        tensor.requires_grad_(requires_grad)
        state[name] = tensor
    return state


def _state_components_and_gradients(
    *,
    ctx: Mapping[str, Any],
    initial_embeddings: Mapping[str, np.ndarray],
    final_embeddings: Mapping[str, np.ndarray],
    initial_residuals: Mapping[str, np.ndarray],
    final_residuals: Mapping[str, np.ndarray],
    masked_nodes: Sequence[str],
    wt: Callable[[tuple[str, str]], Any],
    nw: Mapping[str, Any] | None,
    lam_zero: float,
    lam_norm: float,
    mask_values: Sequence[int],
    near_zero_threshold: float,
    exploding_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate initial/final terms and post-optimization node gradients."""

    import torch

    device = ctx["device"]
    free_names = tuple(ctx["free"])
    residual_names = tuple(ctx["gen_nodes"]) if ctx["use_res"] else ()

    initial_p = _tensor_state(
        initial_embeddings,
        free_names,
        torch=torch,
        device=device,
        requires_grad=False,
    )
    final_p = _tensor_state(
        final_embeddings,
        free_names,
        torch=torch,
        device=device,
        requires_grad=True,
    )
    initial_r = (
        _tensor_state(
            initial_residuals,
            residual_names,
            torch=torch,
            device=device,
            requires_grad=False,
        )
        if residual_names
        else None
    )
    final_r = (
        _tensor_state(
            final_residuals,
            residual_names,
            torch=torch,
            device=device,
            requires_grad=True,
        )
        if residual_names
        else None
    )
    anchors = ctx["At"]

    def initial_emb(node: str):
        return anchors[node] if node in anchors else initial_p[node]

    def final_emb(node: str):
        return anchors[node] if node in anchors else final_p[node]

    initial_components = component_losses(
        ctx,
        initial_emb,
        wt,
        initial_p,
        initial_r,
        lam_zero,
        lam_norm,
        nw=nw,
    )
    final_components = component_losses(
        ctx,
        final_emb,
        wt,
        final_p,
        final_r,
        lam_zero,
        lam_norm,
        nw=nw,
    )
    targets = [final_p[node] for node in masked_nodes]
    total = _active_loss(
        final_components,
        mask_values,
        list(final_p.values()) + ([] if final_r is None else list(final_r.values())),
    )
    total_grads = torch.autograd.grad(
        total, targets, retain_graph=True, allow_unused=True
    )
    total_norm_by_node = {
        node: (
            0.0
            if gradient is None
            else float(gradient.detach().norm().cpu())
        )
        for node, gradient in zip(masked_nodes, total_grads)
    }

    term_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    for active, term in zip(mask_values, TERM_NAMES):
        initial_value = float(initial_components[term].detach().cpu())
        final_value = float(final_components[term].detach().cpu())
        active_initial = initial_value if active else 0.0
        active_final = final_value if active else 0.0
        term_rows.append(
            {
                "term": term,
                "term_active": bool(active),
                "raw_initial_loss": initial_value,
                "raw_final_loss": final_value,
                "raw_loss_delta": final_value - initial_value,
                "active_initial_loss": active_initial,
                "active_final_loss": active_final,
                "active_loss_delta": active_final - active_initial,
                "nonfinite": not all(
                    math.isfinite(value)
                    for value in (
                        initial_value,
                        final_value,
                        active_initial,
                        active_final,
                    )
                ),
            }
        )
        value = final_components[term]
        if value.requires_grad:
            raw_grads = torch.autograd.grad(
                value, targets, retain_graph=True, allow_unused=True
            )
        else:
            raw_grads = (None,) * len(targets)
        for node, raw_gradient in zip(masked_nodes, raw_grads):
            raw_norm = (
                0.0
                if raw_gradient is None
                else float(raw_gradient.detach().norm().cpu())
            )
            active_norm = raw_norm if active else 0.0
            total_norm = total_norm_by_node[node]
            gradient_rows.append(
                {
                    "node_id": node,
                    "term": term,
                    "term_active": bool(active),
                    "raw_final_gradient_norm": raw_norm,
                    "active_final_gradient_norm": active_norm,
                    "total_final_gradient_norm": total_norm,
                    "raw_near_zero": raw_norm <= near_zero_threshold,
                    "near_zero": active_norm <= near_zero_threshold,
                    "exploding": active_norm > exploding_threshold,
                    "nonfinite": not all(
                        math.isfinite(value)
                        for value in (raw_norm, active_norm, total_norm)
                    ),
                }
            )
    return term_rows, gradient_rows


def solve_audit(
    frozen_solver: Any,
    graph: Any,
    weights: Mapping[tuple[str, str], float],
    labeled_embeddings: Mapping[str, np.ndarray],
    d: int,
    *,
    common_initial: CommonInitialState,
    term_mask: str | Mapping[str, Any] | Sequence[Any] = "full_oracle",
    masked_nodes: Sequence[str] | None = None,
    weight_module: Any = None,
    K: int = 60,
    inner_lr: float = 2e-2,
    lam_zero: float = 0.3,
    lam_norm: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    residual: float = 0.0,
    lam_res: float = 0.0,
    partial_corr: Any = None,
    neg_op: Any = None,
    bridge: Any = None,
    feats: Any = None,
    canonical_full_path: bool = False,
    near_zero_threshold: float = 1e-10,
    exploding_threshold: float = 1e3,
    raise_on_nonfinite: bool = True,
) -> AuditSolveResult:
    """Run the frozen functional-Adam dynamics under an explicit term mask.

    ``canonical_full_path=True`` is permitted only for the full mask.  It calls
    the original ``step_loss`` on every update; component evaluation is then
    used only for post-hoc diagnostics.
    """

    import torch

    mask_values = normalize_term_mask(term_mask)
    if canonical_full_path and mask_values != FULL_MASK:
        raise AuditSolverError("canonical_full_path requires the full objective mask")
    if common_initial.d != d:
        raise AuditSolverError(
            f"common initialization dimension {common_initial.d} != requested {d}"
        )
    if common_initial.seed != seed:
        raise AuditSolverError(
            f"common initialization seed {common_initial.seed} != solver seed {seed}"
        )
    if not isinstance(K, int) or K < 0:
        raise AuditSolverError("K must be a non-negative integer")
    if not np.isfinite(float(inner_lr)) or float(inner_lr) <= 0:
        raise AuditSolverError("inner_lr must be positive and finite")
    if not np.isfinite(float(near_zero_threshold)) or near_zero_threshold < 0:
        raise AuditSolverError("near_zero_threshold must be non-negative and finite")
    if not np.isfinite(float(exploding_threshold)) or exploding_threshold <= 0:
        raise AuditSolverError("exploding_threshold must be positive and finite")

    torch.manual_seed(seed)
    labeled = set(labeled_embeddings)
    free = [node for node in graph.nodes if node not in labeled]
    if tuple(free) != common_initial.free_nodes:
        raise AuditSolverError(
            "transformed arm free-node order differs from the common oracle initialization"
        )
    if set(graph.nodes) != set(common_initial.node_order):
        raise AuditSolverError(
            "transformed arm node set differs from the common oracle initialization"
        )
    if not free:
        raise AuditSolverError("audit decomposition requires at least one free node")

    anchors_np = {
        node: np.asarray(value, dtype=np.float64)
        for node, value in labeled_embeddings.items()
    }
    context = frozen_solver.build_ctx(
        graph,
        dict(weights),
        dict(weights),
        anchors_np,
        free,
        d,
        seed,
        device,
        residual,
        lam_res,
        partial_corr,
        neg_op,
        bridge,
    )
    if context["use_res"]:
        missing_residuals = set(context["gen_nodes"]) - set(common_initial.residuals)
        if missing_residuals:
            raise AuditSolverError(
                f"common residual initialization is missing nodes {sorted(missing_residuals)}"
            )
        context["Rv0"] = {
            node: np.asarray(common_initial.residuals[node], dtype=np.float64).copy()
            for node in context["gen_nodes"]
        }

    nw = weight_module(feats, context) if weight_module is not None else None
    P = _tensor_state(
        common_initial.free_embeddings,
        free,
        torch=torch,
        device=device,
        requires_grad=True,
    )
    Rv = (
        _tensor_state(
            context["Rv0"],
            context["gen_nodes"],
            torch=torch,
            device=device,
            requires_grad=True,
        )
        if context["use_res"]
        else None
    )
    initial_embeddings = {
        node: tensor.detach().cpu().numpy().astype(np.float64).copy()
        for node, tensor in P.items()
    }
    initial_residuals = (
        {
            node: tensor.detach().cpu().numpy().astype(np.float64).copy()
            for node, tensor in Rv.items()
        }
        if Rv is not None
        else {}
    )

    params = list(P.values()) + ([] if Rv is None else list(Rv.values()))
    first_moment = [torch.zeros_like(parameter) for parameter in params]
    second_moment = [torch.zeros_like(parameter) for parameter in params]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    weight_constants = context["wt_const"]

    def edge_weight(edge: tuple[str, str]):
        return weight_constants[edge]

    anchors = context["At"]

    def embedding(node: str):
        return anchors[node] if node in anchors else P[node]

    max_gradient_norm = 0.0
    max_parameter_norm = max(float(value.detach().norm().cpu()) for value in params)
    nonfinite_seen = False
    steps_completed = 0
    last_step_loss: float | None = None

    for step in range(1, K + 1):
        if canonical_full_path:
            loss = frozen_solver.step_loss(
                context,
                embedding,
                edge_weight,
                P,
                Rv,
                lam_zero,
                lam_norm,
                nw=nw,
            )
        else:
            components = component_losses(
                context,
                embedding,
                edge_weight,
                P,
                Rv,
                lam_zero,
                lam_norm,
                nw=nw,
            )
            loss = _active_loss(components, mask_values, params)

        gradients = torch.autograd.grad(
            loss, params, create_graph=False, allow_unused=True
        )
        gradients = tuple(
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(params, gradients)
        )
        loss_value = float(loss.detach().cpu())
        gradient_norm = math.sqrt(
            sum(float((gradient.detach() ** 2).sum().cpu()) for gradient in gradients)
        )
        finite_step = math.isfinite(loss_value) and math.isfinite(gradient_norm)
        finite_step = finite_step and all(
            bool(torch.isfinite(gradient).all().item()) for gradient in gradients
        )
        if not finite_step:
            nonfinite_seen = True
            if raise_on_nonfinite:
                raise NonFiniteOptimizationError(
                    f"non-finite objective or gradient at functional-Adam step {step}"
                )

        updated: list[Any] = []
        for index, (parameter, gradient) in enumerate(zip(params, gradients)):
            first_moment[index] = (
                beta1 * first_moment[index] + (1.0 - beta1) * gradient
            )
            second_moment[index] = (
                beta2 * second_moment[index]
                + (1.0 - beta2) * gradient * gradient
            )
            corrected_first = first_moment[index] / (1.0 - beta1**step)
            corrected_second = second_moment[index] / (1.0 - beta2**step)
            updated.append(
                parameter
                - inner_lr
                * corrected_first
                / (corrected_second.sqrt() + epsilon)
            )

        params = [value.detach().requires_grad_(True) for value in updated]
        first_moment = [value.detach() for value in first_moment]
        second_moment = [value.detach() for value in second_moment]
        offset = 0
        for node in list(P):
            P[node] = params[offset]
            offset += 1
        if Rv is not None:
            for node in list(Rv):
                Rv[node] = params[offset]
                offset += 1

        max_gradient_norm = max(max_gradient_norm, gradient_norm)
        max_parameter_norm = max(
            max_parameter_norm,
            max(float(value.detach().norm().cpu()) for value in params),
        )
        last_step_loss = loss_value
        steps_completed = step

    final_embeddings = {
        node: tensor.detach().cpu().numpy().astype(np.float64)
        for node, tensor in P.items()
    }
    final_residuals = (
        {
            node: tensor.detach().cpu().numpy().astype(np.float64)
            for node, tensor in Rv.items()
        }
        if Rv is not None
        else {}
    )
    if masked_nodes is None:
        masked_nodes = tuple(free)
    else:
        masked_nodes = tuple(masked_nodes)
    if not set(masked_nodes) <= set(free):
        raise AuditSolverError("every masked node must be a free embedding")

    loss_terms, gradient_norms = _state_components_and_gradients(
        ctx=context,
        initial_embeddings=initial_embeddings,
        final_embeddings=final_embeddings,
        initial_residuals=initial_residuals,
        final_residuals=final_residuals,
        masked_nodes=masked_nodes,
        wt=edge_weight,
        nw=nw,
        lam_zero=lam_zero,
        lam_norm=lam_norm,
        mask_values=mask_values,
        near_zero_threshold=near_zero_threshold,
        exploding_threshold=exploding_threshold,
    )

    output = {
        node: np.asarray(value, dtype=np.float64)
        for node, value in labeled_embeddings.items()
    }
    output.update(final_embeddings)
    finite_final = all(np.isfinite(value).all() for value in output.values())
    finite_final = finite_final and all(
        np.isfinite(value).all() for value in final_residuals.values()
    )
    nonfinite_seen = nonfinite_seen or not finite_final
    if nonfinite_seen and raise_on_nonfinite:
        raise NonFiniteOptimizationError("audit solver produced a non-finite final state")

    return AuditSolveResult(
        embeddings=output,
        initial_embeddings=initial_embeddings,
        final_embeddings=final_embeddings,
        initial_residuals=initial_residuals,
        final_residuals=final_residuals,
        loss_terms=loss_terms,
        gradient_norms=gradient_norms,
        trace={
            "steps_requested": K,
            "steps_completed": steps_completed,
            "last_preupdate_loss": last_step_loss,
            "max_total_gradient_norm": max_gradient_norm,
            "max_parameter_norm": max_parameter_norm,
            "nonfinite_seen": nonfinite_seen,
            "near_zero_threshold": float(near_zero_threshold),
            "exploding_threshold": float(exploding_threshold),
        },
        term_mask={
            name: bool(value) for name, value in zip(TERM_NAMES, mask_values)
        },
        canonical_full_path=bool(canonical_full_path),
    )


def parity_summary(
    reference: Mapping[str, np.ndarray],
    audit: Mapping[str, np.ndarray],
    nodes: Sequence[str],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    max_cosine_error: float = 1e-7,
) -> dict[str, Any]:
    """Compare a canonical audit solve with ``solve_unrolled`` output."""

    if not nodes:
        raise AuditSolverError("parity comparison requires at least one node")
    ref = np.stack([np.asarray(reference[node], dtype=np.float64) for node in nodes])
    got = np.stack([np.asarray(audit[node], dtype=np.float64) for node in nodes])
    ref_lengths = np.linalg.norm(ref, axis=1)
    got_lengths = np.linalg.norm(got, axis=1)
    cosine_error = np.empty(len(nodes), dtype=np.float64)
    both_zero = (ref_lengths <= 1e-12) & (got_lengths <= 1e-12)
    one_zero = (ref_lengths <= 1e-12) ^ (got_lengths <= 1e-12)
    regular = ~(both_zero | one_zero)
    cosine_error[both_zero] = 0.0
    cosine_error[one_zero] = 1.0
    if np.any(regular):
        cosine = np.sum(ref[regular] * got[regular], axis=1) / (
            ref_lengths[regular] * got_lengths[regular]
        )
        cosine_error[regular] = 1.0 - cosine
    max_abs = float(np.max(np.abs(ref - got)))
    max_cos = float(np.max(np.abs(cosine_error)))
    allclose = bool(np.allclose(ref, got, rtol=rtol, atol=atol))
    passed = allclose and max_cos <= max_cosine_error
    return {
        "passed": passed,
        "allclose": allclose,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_difference": max_abs,
        "max_cosine_error": max_cos,
        "max_allowed_cosine_error": float(max_cosine_error),
        "nodes": len(nodes),
    }


__all__ = [
    "AuditSolveResult",
    "AuditSolverError",
    "CommonInitialState",
    "FULL_MASK",
    "NonFiniteOptimizationError",
    "TERM_MASKS",
    "TERM_NAMES",
    "component_losses",
    "make_common_initial_state",
    "normalize_term_mask",
    "parity_summary",
    "solve_audit",
    "term_mask_dict",
]
