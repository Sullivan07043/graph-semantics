"""Training-free external baselines for graph-semantic completion.

This module deliberately contains no runner or decoding logic.  Both baselines
return node-name -> numpy-vector dictionaries so Task 1 and Task 2 can share the
same encoder, dictionary, decoder, and evaluation code as the main method.

Feature Propagation follows the update in Rossi et al. (2022): missing features
are diffused with the symmetrically normalized adjacency while observed
features are clamped after every iteration.  ``Graph`` edges are directed, but
Feature Propagation is defined here on their binary undirected projection.  It
is therefore an explicit causal-graph adaptation, not an unmodified run of the
authors' homogeneous undirected-graph benchmark code.
"""
from collections.abc import Mapping

import numpy as np


FEATURE_PROP_VERSION = "rossi22-fp-undirected-causal-projection-v1"
LOADING_CENTROID_VERSION = "pc1-loading-centroid-measurement-dag-v1"
MB_CONTEXT_VERSION = "visible-typed-markov-blanket-v1"


def _embedding_dict(embeddings, valid_nodes, name):
    """Validate and copy a sparse node -> 1-D embedding mapping."""
    if not isinstance(embeddings, Mapping):
        raise TypeError(f"{name} must be a node -> embedding mapping")

    valid = set(valid_nodes)
    unknown = set(embeddings) - valid
    if unknown:
        raise KeyError(f"{name} contains nodes outside the graph: {sorted(unknown)}")
    if not embeddings:
        raise ValueError(f"{name} must contain at least one embedding")

    out, dim = {}, None
    for node, value in embeddings.items():
        vector = np.asarray(value, dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"embedding for {node!r} must be a non-empty 1-D vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"embedding for {node!r} contains a non-finite value")
        if dim is None:
            dim = vector.size
        elif vector.size != dim:
            raise ValueError(
                f"embedding for {node!r} has dimension {vector.size}; expected {dim}"
            )
        out[node] = vector.copy()
    return out, dim


def _fallback_vector(embeddings, dim, fallback):
    """Resolve a common fallback for nodes with no visible semantic anchor."""
    if isinstance(fallback, str):
        if fallback == "mean":
            return np.mean(np.stack(list(embeddings.values())), axis=0)
        if fallback == "zeros":
            return np.zeros(dim, dtype=float)
        raise ValueError("fallback must be 'mean', 'zeros', or a 1-D vector")

    vector = np.asarray(fallback, dtype=float)
    if vector.ndim != 1 or vector.size != dim:
        raise ValueError(f"fallback vector must have shape ({dim},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("fallback vector contains a non-finite value")
    return vector.copy()


def feature_propagation(
    graph,
    known_embeddings,
    *,
    max_iter=40,
    tol=1e-8,
    fallback="zeros",
):
    """Recover missing node embeddings with clamped Feature Propagation.

    Parameters
    ----------
    graph
        Existing :class:`graph.Graph` instance.  Directed edges are converted
        to a binary undirected adjacency before propagation.
    known_embeddings
        Sparse mapping from any known graph nodes to their embeddings.  These
        vectors are copied back exactly after every diffusion step.
    max_iter, tol
        Iteration limit and absolute convergence tolerance.  The 40-step
        default matches the standard small-graph Feature Propagation setup.
    fallback
        ``"zeros"`` (default), ``"mean"``, or an explicit vector.  It is used
        for every connected component that contains no known node, including
        isolated nodes, because propagation cannot identify such a component.

    Returns
    -------
    dict
        Embeddings for every node in ``graph.nodes``, in graph node order.
    """
    nodes = list(graph.nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError("graph.nodes must contain unique node names")
    if not nodes:
        if known_embeddings:
            raise KeyError("known_embeddings contains nodes but the graph is empty")
        return {}
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite non-negative number")

    known, dim = _embedding_dict(known_embeddings, nodes, "known_embeddings")
    fill = _fallback_vector(known, dim, fallback)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # Binary undirected projection: reciprocal directed edges must not receive
    # twice the influence of a single edge.
    adjacency = np.zeros((n, n), dtype=float)
    for source, target in graph.edges:
        i, j = index[source], index[target]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree)
    nonisolated = degree > 0
    inv_sqrt_degree[nonisolated] = 1.0 / np.sqrt(degree[nonisolated])
    normalized = (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )

    known_indices = np.asarray([index[node] for node in known], dtype=int)
    known_values = np.stack([known[node] for node in known])
    values = np.zeros((n, dim), dtype=float)
    values[known_indices] = known_values

    # Mark every undirected component reachable from a known semantic anchor.
    # Components without an anchor remain mathematically unidentifiable under
    # Feature Propagation and are filled explicitly after the iterations.
    anchored = np.zeros(n, dtype=bool)
    anchored[known_indices] = True
    frontier = anchored.copy()
    while frontier.any():
        reached = (adjacency @ frontier.astype(float)) > 0
        new_frontier = reached & ~anchored
        if not new_frontier.any():
            break
        anchored |= new_frontier
        frontier = new_frontier

    for _ in range(max_iter):
        updated = normalized @ values
        updated[known_indices] = known_values
        if np.max(np.abs(updated - values)) <= tol:
            values = updated
            break
        values = updated

    values[~anchored] = fill
    # Assignment above cannot alter known rows, but copy them once more to make
    # the clamp contract explicit even under unusual floating-point inputs.
    values[known_indices] = known_values
    return {node: values[i].copy() for i, node in enumerate(nodes)}


def _edge_loading(loadings, latent, child):
    """Read a scalar loading from edge, nested, or child-score mappings."""
    if loadings is None:
        return None

    value = None
    if (latent, child) in loadings:
        value = loadings[(latent, child)]
    elif latent in loadings and isinstance(loadings[latent], Mapping):
        value = loadings[latent].get(child)
    elif child in loadings:
        value = loadings[child]

    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 0:
        return None
    value = float(array)
    return value if np.isfinite(value) else None


def loading_centroid_applicability(graph):
    """Return whether the PC1-loading baseline has a measurement-DAG meaning.

    Latent-to-latent hierarchy is allowed (for example HEXACO factor -> facet ->
    item).  An observed source edge is not: once an observed variable causes a
    latent or another observed variable, direct children/descendants are no
    longer a conventional indicator set.  TLVD is such a general DAG and must
    use a separately named structural baseline instead of this comparator.
    """
    observed = set(graph.observed)
    observed_source_edges = [
        (source, target)
        for source, target in graph.edges
        if source in observed
    ]
    if observed_source_edges:
        return False, (
            "observed-source edges make indicator loadings ambiguous "
            f"({len(observed_source_edges)} edge(s))"
        )
    return True, "latent-only sources form a measurement DAG"


def loading_centroid(
    graph,
    visible_embeddings,
    W=None,
    *,
    score=None,
    fallback="mean",
):
    """Build one embedding per latent from its visible observed children.

    Direct observed children are preferred.  If none are visible, all visible
    observed descendants are used.  The centroid weights are ``abs(loading)``:
    ``W`` may be an edge mapping such as the first output of
    ``Graph.estimate_weights``.  Alternatively, ``score`` may provide scalar
    loadings as ``(latent, child) -> value``, ``latent -> {child: value}``, or
    ``child -> value``.  ``W`` takes precedence when both are supplied.

    Missing/non-finite weights count as zero.  If no candidate has positive
    absolute weight, candidates receive uniform weight.  A latent with no
    visible observed descendant receives the common ``fallback`` (the global
    visible-embedding mean by default).
    """
    applicable, reason = loading_centroid_applicability(graph)
    if not applicable:
        raise ValueError(
            "loading_centroid is only defined for measurement DAGs: " + reason
        )

    nodes = list(graph.nodes)
    visible, dim = _embedding_dict(
        visible_embeddings, nodes, "visible_embeddings"
    )
    fill = _fallback_vector(visible, dim, fallback)
    observed = set(graph.observed)
    loadings = W if W is not None else score
    if loadings is not None and not isinstance(loadings, Mapping):
        raise TypeError("W/score must be a loading mapping")

    out = {}
    for latent in graph.latents:
        direct = [
            child
            for child in graph.children(latent)
            if child in observed and child in visible
        ]
        candidates = direct
        if not candidates:
            candidates = [
                child
                for child in graph.observed_descendants(latent)
                if child in visible
            ]

        if not candidates:
            out[latent] = fill.copy()
            continue

        vectors = np.stack([visible[child] for child in candidates])
        weights = np.asarray(
            [
                abs(value) if (value := _edge_loading(loadings, latent, child))
                is not None else 0.0
                for child in candidates
            ],
            dtype=float,
        )
        if weights.sum() <= 0:
            weights.fill(1.0)
        out[latent] = (weights / weights.sum()) @ vectors

    return out


def latent_markov_context(graph, latent, labels, visible_nodes):
    """Build a leakage-safe, typed Markov-blanket context for latent naming.

    Only descriptions belonging to ``visible_nodes`` are returned.  Latent
    neighbours are represented by counts rather than node names because several
    benchmark graphs use the gold construct name as the latent node identifier.
    Hidden observed-node names are likewise never exposed.
    """
    if latent not in set(graph.latents):
        raise KeyError(f"{latent!r} is not a latent node")
    visible = set(visible_nodes)
    observed = set(graph.observed)
    unknown_visible = visible - observed
    if unknown_visible:
        raise KeyError(
            "visible_nodes contains non-observed nodes: "
            + ", ".join(sorted(unknown_visible))
        )

    parents = set(graph.parents(latent))
    children = set(graph.children(latent))
    spouses = set(graph.markov_blanket(latent)) - parents - children

    def relation(nodes):
        visible_observed = sorted(
            str(labels[node])
            for node in nodes
            if node in observed and node in visible
        )
        hidden_observed = sum(
            node in observed and node not in visible for node in nodes
        )
        anonymous_latents = sum(node not in observed for node in nodes)
        return {
            "visible_observed_labels": visible_observed,
            "hidden_observed_count": int(hidden_observed),
            "anonymous_latent_count": int(anonymous_latents),
        }

    return {
        "version": MB_CONTEXT_VERSION,
        "parents": relation(parents),
        "children": relation(children),
        "spouses": relation(spouses),
    }


def format_latent_markov_context(context):
    """Render :func:`latent_markov_context` without introducing node names."""
    role_names = {
        "parents": "Parents of the target latent",
        "children": "Children of the target latent",
        "spouses": "Other parents of its children (spouses)",
    }
    lines = []
    for role in ("parents", "children", "spouses"):
        item = context[role]
        lines.append(f"{role_names[role]}:")
        labels = item["visible_observed_labels"]
        if labels:
            lines.extend(f"- visible observed measure: {label}" for label in labels)
        if item["hidden_observed_count"]:
            lines.append(
                f"- {item['hidden_observed_count']} observed measure label(s) hidden by this fold"
            )
        if item["anonymous_latent_count"]:
            lines.append(
                f"- {item['anonymous_latent_count']} anonymous latent node(s)"
            )
        if (
            not labels
            and not item["hidden_observed_count"]
            and not item["anonymous_latent_count"]
        ):
            lines.append("- none")
    return "\n".join(lines)
