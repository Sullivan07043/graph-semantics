#!/usr/bin/env python3
"""Executable A -> B -> C orientation audit for Task 3 E0''.

This diagnostic deliberately calls the existing E0' adapter and the frozen
Stage-3 graph/loss implementations.  It does not train, tune, or modify any
artifact.  The default output is a structured JSON record on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task3_v2.scripts import e0_core  # noqa: E402
from task3_v2.scripts import run_e0_bridge  # noqa: E402
from v5 import graph as stage3_graph  # noqa: E402
from v5 import l2_solver as stage3_solver  # noqa: E402
from v5 import optimize as stage3_optimize  # noqa: E402


NODE_IDS = ("A", "B", "C")
EXPECTED_EDGE_PAIRS = (("A", "B"), ("B", "C"))
EDGE_COEFFICIENTS = {("A", "B"): 0.7, ("B", "C"): 0.4}
EXPECTED_PARENTS = {"A": [], "B": ["A"], "C": ["B"]}
EXPECTED_CHILDREN = {"A": ["B"], "B": ["C"], "C": []}
EXPECTED_ADJACENCY = np.asarray(
    [
        [0.0, 0.7, 0.0],
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
FIXED_EMBEDDINGS = {
    "A": np.asarray([1.0, 0.0], dtype=np.float64),
    "C": np.asarray([0.0, 1.0], dtype=np.float64),
}


class OrientationAuditError(AssertionError):
    """Raised when any frozen source/target orientation assertion fails."""


def build_chain_spec(*, transpose: bool = False) -> dict[str, Any]:
    """Build the minimal weighted chain fixture.

    ``transpose=True`` is a deliberate negative control.  It retains the edge
    coefficients but reverses both ordered endpoint pairs.
    """

    edges = []
    for source, target in EXPECTED_EDGE_PAIRS:
        if transpose:
            source, target = target, source
        coefficient = EDGE_COEFFICIENTS[
            (target, source) if transpose else (source, target)
        ]
        edges.append(
            {
                "source": source,
                "target": target,
                "coefficient": coefficient,
            }
        )
    return {
        "nodes": [{"id": node_id} for node_id in NODE_IDS],
        "edges": edges,
    }


def edge_pairs(spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Use the actual E0' JSON-to-Graph edge adapter."""

    return run_e0_bridge._edge_pairs(spec)


def source_by_target_adjacency(spec: Mapping[str, Any]) -> np.ndarray:
    """Use the actual E0' source-row/target-column adjacency helper."""

    return e0_core.adjacency_matrix(
        spec,
        weighted=True,
        validate_design_constraints=False,
    )


def _deterministic_fit_data() -> np.ndarray:
    """Non-degenerate observed-only data used only to exercise the adapter."""

    source = np.linspace(-2.0, 2.0, 101, dtype=np.float64)
    middle = 0.7 * source + 0.1 * np.sin(3.0 * source)
    target = 0.4 * middle + 0.1 * np.cos(5.0 * source)
    return np.column_stack([source, middle, target])


def build_adapter_context(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run the exact E0' observed-only Graph adapter with frozen true weights."""

    true_weights = {
        (str(edge["source"]), str(edge["target"])): float(edge["coefficient"])
        for edge in spec["edges"]
    }
    return run_e0_bridge._make_graph_context(
        {"graph": stage3_graph, "optimize": stage3_optimize},
        spec,
        _deterministic_fit_data(),
        {},
        true_weights=true_weights,
    )


def _raise_mismatch(name: str, actual: Any, expected: Any) -> None:
    raise OrientationAuditError(f"{name} mismatch: actual={actual!r}, expected={expected!r}")


def assert_expected_chain_orientation(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless every JSON, adjacency, and Graph convention agrees."""

    actual_pairs = edge_pairs(spec)
    if actual_pairs != list(EXPECTED_EDGE_PAIRS):
        _raise_mismatch("ordered edge pairs", actual_pairs, list(EXPECTED_EDGE_PAIRS))

    adjacency = source_by_target_adjacency(spec)
    if not np.array_equal(adjacency, EXPECTED_ADJACENCY):
        _raise_mismatch(
            "source-by-target adjacency",
            adjacency.tolist(),
            EXPECTED_ADJACENCY.tolist(),
        )

    adapter_context = build_adapter_context(spec)
    graph = adapter_context["graph"]
    parents = {node_id: graph.parents(node_id) for node_id in NODE_IDS}
    children = {node_id: graph.children(node_id) for node_id in NODE_IDS}
    if parents != EXPECTED_PARENTS:
        _raise_mismatch("Graph.parents", parents, EXPECTED_PARENTS)
    if children != EXPECTED_CHILDREN:
        _raise_mismatch("Graph.children", children, EXPECTED_CHILDREN)
    if list(graph.edges) != list(EXPECTED_EDGE_PAIRS):
        _raise_mismatch("adapter Graph.edges", graph.edges, EXPECTED_EDGE_PAIRS)
    if set(adapter_context["weights"]) != set(EXPECTED_EDGE_PAIRS):
        _raise_mismatch(
            "adapter weight keys",
            sorted(adapter_context["weights"]),
            sorted(EXPECTED_EDGE_PAIRS),
        )
    adapter_weights = {
        edge: float(adapter_context["weights"][edge])
        for edge in EXPECTED_EDGE_PAIRS
    }
    if adapter_weights != EDGE_COEFFICIENTS:
        _raise_mismatch(
            "adapter weight values",
            adapter_weights,
            EDGE_COEFFICIENTS,
        )

    return {
        "json_edge_pairs": [list(pair) for pair in actual_pairs],
        "adjacency_convention": "row=source,parent; column=target,child",
        "adjacency": adjacency.tolist(),
        "adapter_edges": [list(pair) for pair in graph.edges],
        "adapter_weight_keys": [list(pair) for pair in adapter_context["weights"]],
        "adapter_weights": {
            f"{source}->{target}": value
            for (source, target), value in adapter_weights.items()
        },
        "parents": parents,
        "children": children,
    }


def _build_solver_context(
    edge_pairs_: Sequence[tuple[str, str]],
    weights: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    graph = stage3_graph.Graph([], list(NODE_IDS), list(edge_pairs_))
    return stage3_solver.build_ctx(
        graph,
        dict(weights),
        dict(weights),
        dict(FIXED_EMBEDDINGS),
        ["B"],
        2,
        0,
        "cpu",
        0.0,
        0.0,
        None,
        None,
        None,
    )


def generation_roles(context: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    """Parse the incoming and outgoing generation equations containing a node."""

    incoming = [
        (parent, node_id)
        for parent in context["parents"].get(node_id, [])
    ]
    outgoing = [
        (node_id, child)
        for child in context["gen_nodes"]
        if node_id in context["parents"][child]
    ]
    return {
        "incoming_edges": [list(pair) for pair in incoming],
        "outgoing_edges": [list(pair) for pair in outgoing],
        "generated_nodes": list(context["gen_nodes"]),
        "generation_parents": {
            node: list(parents) for node, parents in context["parents"].items()
        },
    }


def _generation_loss_and_gradient(
    edge_pairs_: Sequence[tuple[str, str]],
    weights: Mapping[tuple[str, str], float],
    z_b: Sequence[float] = (0.0, 0.0),
) -> tuple[float, np.ndarray]:
    """Evaluate the exact frozen Stage-3 generation loss and dL/dz_B."""

    import torch

    context = _build_solver_context(edge_pairs_, weights)
    b = torch.tensor(tuple(z_b), dtype=torch.float32, requires_grad=True)
    free_embeddings = {"B": b}

    def embedding(node_id: str):
        return (
            context["At"][node_id]
            if node_id in context["At"]
            else free_embeddings[node_id]
        )

    def weight(edge: tuple[str, str]):
        return context["wt_const"][edge]

    loss = stage3_solver.step_loss(
        context,
        embedding,
        weight,
        free_embeddings,
        None,
        0.0,
        0.0,
        nw=None,
    )
    gradient = torch.autograd.grad(loss, b)[0]
    return (
        float(loss.detach().cpu()),
        gradient.detach().cpu().numpy().astype(np.float64),
    )


def generation_audit() -> dict[str, Any]:
    """Verify the adapter-fed incoming/outgoing contributions to dL/dz_B."""

    adapter_context = build_adapter_context(build_chain_spec())
    adapter_edges = list(adapter_context["graph"].edges)
    adapter_weights = {
        edge: float(adapter_context["weights"][edge])
        for edge in adapter_edges
    }
    full_context = _build_solver_context(adapter_edges, adapter_weights)
    roles = generation_roles(full_context, "B")
    expected_roles = {
        "incoming_edges": [["A", "B"]],
        "outgoing_edges": [["B", "C"]],
        "generated_nodes": ["B", "C"],
        "generation_parents": {"B": ["A"], "C": ["B"]},
    }
    if roles != expected_roles:
        _raise_mismatch("generation roles", roles, expected_roles)

    full_loss, full_gradient = _generation_loss_and_gradient(
        adapter_edges,
        adapter_weights,
    )
    incoming_loss, incoming_gradient = _generation_loss_and_gradient(
        [("A", "B")],
        {("A", "B"): adapter_weights[("A", "B")]},
    )
    outgoing_loss, outgoing_gradient = _generation_loss_and_gradient(
        [("B", "C")],
        {("B", "C"): adapter_weights[("B", "C")]},
    )
    expected_gradient = np.asarray([-1.4, -0.8], dtype=np.float64)
    if not np.isclose(full_loss, 1.49, atol=1e-7, rtol=0.0):
        _raise_mismatch("generation loss", full_loss, 1.49)
    if not np.allclose(full_gradient, expected_gradient, atol=1e-7, rtol=0.0):
        _raise_mismatch(
            "dL/dz_B",
            full_gradient.tolist(),
            expected_gradient.tolist(),
        )
    if not np.allclose(
        full_gradient,
        incoming_gradient + outgoing_gradient,
        atol=1e-7,
        rtol=0.0,
    ):
        _raise_mismatch(
            "incoming + outgoing gradient",
            (incoming_gradient + outgoing_gradient).tolist(),
            full_gradient.tolist(),
        )

    return {
        **roles,
        "z_A": FIXED_EMBEDDINGS["A"].tolist(),
        "z_B_at_evaluation": [0.0, 0.0],
        "z_C": FIXED_EMBEDDINGS["C"].tolist(),
        "weight_A_to_B": adapter_weights[("A", "B")],
        "weight_B_to_C": adapter_weights[("B", "C")],
        "expected_full_loss": 1.49,
        "full_loss": full_loss,
        "incoming_only_loss": incoming_loss,
        "outgoing_only_loss": outgoing_loss,
        "incoming_gradient": incoming_gradient.tolist(),
        "outgoing_gradient": outgoing_gradient.tolist(),
        "gradient_dL_dz_B": full_gradient.tolist(),
        "expected_gradient_dL_dz_B": expected_gradient.tolist(),
        "negative_gradient_descent_direction": (-full_gradient).tolist(),
        "interpretation": "z_B is pulled toward both its parent z_A and its child z_C",
    }


def als_audit() -> dict[str, Any]:
    """Compare the frozen ALS initialization with its ridge-aware closed form."""

    graph = stage3_graph.Graph([], list(NODE_IDS), list(EXPECTED_EDGE_PAIRS))
    actual = stage3_optimize._solve_embeddings(
        graph,
        dict(EDGE_COEFFICIENTS),
        dict(FIXED_EMBEDDINGS),
        ["B"],
        2,
    )["B"]
    a = EDGE_COEFFICIENTS[("A", "B")]
    b = EDGE_COEFFICIENTS[("B", "C")]
    ideal = (a * FIXED_EMBEDDINGS["A"] + b * FIXED_EMBEDDINGS["C"]) / (
        1.0 + b**2
    )
    implemented = (
        a * FIXED_EMBEDDINGS["A"] + b * FIXED_EMBEDDINGS["C"]
    ) / (1.0 + b**2 + 1e-6)
    if not np.allclose(actual, implemented, atol=1e-12, rtol=0.0):
        _raise_mismatch(
            "ridge-aware ALS solution",
            actual.tolist(),
            implemented.tolist(),
        )
    return {
        "frozen_solver_value": actual.tolist(),
        "ideal_unregularized_closed_form": ideal.tolist(),
        "implemented_ridge_closed_form": implemented.tolist(),
        "ridge": 1e-6,
        "max_abs_error_vs_implemented": float(np.max(np.abs(actual - implemented))),
    }


def reverse_symmetry_audit() -> dict[str, Any]:
    """Demonstrate the generation factor's exact unit-weight reversal symmetry."""

    point = (0.25, -0.3)
    forward_edges = [("A", "B"), ("B", "C")]
    reversed_edges = [("B", "A"), ("C", "B")]
    forward_weights = {edge: 1.0 for edge in forward_edges}
    reversed_weights = {edge: 1.0 for edge in reversed_edges}
    forward_loss, forward_gradient = _generation_loss_and_gradient(
        forward_edges,
        forward_weights,
        point,
    )
    reversed_loss, reversed_gradient = _generation_loss_and_gradient(
        reversed_edges,
        reversed_weights,
        point,
    )
    if not np.isclose(forward_loss, reversed_loss, atol=1e-7, rtol=0.0):
        _raise_mismatch("unit-weight reverse loss", reversed_loss, forward_loss)
    if not np.allclose(
        forward_gradient,
        reversed_gradient,
        atol=1e-7,
        rtol=0.0,
    ):
        _raise_mismatch(
            "unit-weight reverse gradient",
            reversed_gradient.tolist(),
            forward_gradient.tolist(),
        )
    return {
        "z_B_at_evaluation": list(point),
        "forward_loss": forward_loss,
        "reversed_loss": reversed_loss,
        "forward_gradient": forward_gradient.tolist(),
        "reversed_gradient": reversed_gradient.tolist(),
        "exactly_equal_within_tolerance": True,
        "interpretation": (
            "each generation equation is a bidirectional quadratic compatibility "
            "factor during joint embedding completion"
        ),
    }


def run_audit() -> dict[str, Any]:
    """Run every positive and negative-control assertion."""

    orientation = assert_expected_chain_orientation(build_chain_spec())
    generation = generation_audit()
    als = als_audit()
    reverse_symmetry = reverse_symmetry_audit()

    transpose_rejected = False
    transpose_error = ""
    try:
        assert_expected_chain_orientation(build_chain_spec(transpose=True))
    except OrientationAuditError as exc:
        transpose_rejected = True
        transpose_error = str(exc)
    if not transpose_rejected:
        raise OrientationAuditError(
            "deliberately transposed A <- B <- C fixture was not rejected"
        )

    return {
        "schema_version": 1,
        "audit": "Task 3 E0'' orientation interface audit",
        "status": "passed",
        "orientation": orientation,
        "generation": generation,
        "als": als,
        "unit_weight_reverse_symmetry": reverse_symmetry,
        "negative_control": {
            "fixture": "deliberately transposed A <- B <- C",
            "rejected": transpose_rejected,
            "error": transpose_error,
        },
        "verdict": {
            "orientation_interface_bug": False,
            "rerun_e0_prime_for_orientation_fix": False,
            "finding": (
                "source/target conventions are correct; the frozen generation "
                "energy propagates gradients through both endpoints"
            ),
            "next_diagnostic": (
                "continue frozen constraint decomposition without changing or "
                "rerunning E0' for an orientation repair"
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the same structured JSON emitted on stdout.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact rather than indented JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_audit()
        exit_code = 0
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "audit": "Task 3 E0'' orientation interface audit",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 1
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        sort_keys=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
