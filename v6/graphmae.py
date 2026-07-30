"""Small-graph GraphMAE baseline for semantic embedding reconstruction.

This module deliberately has no dataset or encoder imports.  Callers supply:

* graph-like objects exposing ``nodes``, ``observed``, ``latents``, and ``edges``;
* complete frozen-encoder embeddings for the observed nodes of each *development*
  graph used for training; and
* the visible embeddings for a test fold at inference time.

Consequently, fitting cannot discover or load held-out labels by accident.  The
checkpoint contains hyperparameters, model weights, and audit metadata--never
graph examples, label text, semantic embeddings, or dataset paths.

The model follows the useful parts of GraphMAE for this setting: semantic features
are replaced by a learned mask token, a dense normalized-adjacency GCN encodes the
typed graph, target representations are re-masked before a GCN decoder, and masked
observed nodes are trained with scaled cosine error (SCE).  Dense adjacency is
intentional: the benchmark contains many small graphs, for which it avoids a DGL
or PyG dependency and keeps the implementation auditable.

Latent nodes have no supervised name embedding.  During training they are always
feature-masked, but their encoder representation is retained for decoding.  Their
inference embedding is therefore a structural, neighbour-conditioned prediction
through the decoder shared with supervised observed nodes; no latent pseudo-label
or hidden gold name is used.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ArrayLike = Union[np.ndarray, Tensor, Sequence[float]]
CHECKPOINT_FORMAT = "graphmae-small-graph-v1"
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "graphmae.pt")


def file_sha256(path, chunk_size=1024 * 1024):
    """Return a stable content fingerprint for an encoder/checkpoint artifact."""
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GraphExample:
    """One caller-owned development example.

    ``observed_embeddings`` must contain every node in ``graph.observed`` and no
    other nodes.  This strict contract prevents a latent gold name from entering
    training through a permissive feature dictionary.
    """

    graph: Any
    observed_embeddings: Mapping[str, ArrayLike]


@dataclass(frozen=True)
class GraphMAEConfig:
    """Hyperparameters for :class:`GraphMAEBaseline`.

    Defaults are intentionally modest for collections of small causal graphs.
    The graph is symmetrized by default because both parent-to-child and
    child-to-parent semantic messages are needed when latent nodes have no input
    labels.  Set ``undirected=False`` only for a directed ablation.
    """

    hidden_dim: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.0
    mask_rate: float = 0.5
    masks_per_graph: int = 1
    loss_alpha: float = 2.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    grad_clip: float = 1.0
    use_type_features: bool = True
    undirected: bool = True
    self_loops: bool = True
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    seed: int = 0
    deterministic: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0.0 < self.mask_rate <= 1.0:
            raise ValueError("mask_rate must lie in (0, 1]")
        if self.masks_per_graph < 1:
            raise ValueError("masks_per_graph must be positive")
        if self.loss_alpha <= 0.0:
            raise ValueError("loss_alpha must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.grad_clip < 0.0:
            raise ValueError("grad_clip must be non-negative")


@dataclass
class _PreparedGraph:
    graph: Any
    node_names: tuple[str, ...]
    node_index: dict[str, int]
    observed_indices: Tensor
    latent_mask: Tensor
    node_types: Tensor
    features: Tensor
    adjacency: Tensor


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {spec!r}")
    return device


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Dense matmul/linear kernels used below have deterministic variants on the
        # supported devices.  warn_only avoids making this small baseline unusable
        # on an older CUDA/PyTorch combination with an unregistered kernel.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _graph_parts(graph: Any) -> tuple[tuple[str, ...], set[str], set[str]]:
    required = ("nodes", "observed", "latents", "edges")
    missing_attrs = [name for name in required if not hasattr(graph, name)]
    if missing_attrs:
        raise TypeError(f"graph is missing required attributes: {missing_attrs}")

    nodes = tuple(graph.nodes)
    observed = set(graph.observed)
    latents = set(graph.latents)
    node_set = set(nodes)
    if not nodes:
        raise ValueError("graph must contain at least one node")
    if len(node_set) != len(nodes):
        raise ValueError("graph.nodes contains duplicate node names")
    if observed & latents:
        raise ValueError("graph.observed and graph.latents must be disjoint")
    if observed | latents != node_set:
        raise ValueError("graph.nodes must be exactly graph.observed + graph.latents")
    for edge in graph.edges:
        if len(edge) != 2:
            raise ValueError(f"edge must be a (source, target) pair, got {edge!r}")
        source, target = edge
        if source not in node_set or target not in node_set:
            raise ValueError(f"edge refers to a node absent from graph.nodes: {edge!r}")
    return nodes, observed, latents


def normalized_adjacency(
    graph: Any,
    *,
    undirected: bool = True,
    self_loops: bool = True,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build the dense normalized message-passing adjacency for ``graph``.

    Rows are receivers and columns are senders.  With the default symmetrization,
    this is ``D^-1/2 (A + I) D^-1/2``.  Duplicate edges do not change the matrix.
    """

    nodes, _, _ = _graph_parts(graph)
    index = {name: i for i, name in enumerate(nodes)}
    adjacency = torch.zeros((len(nodes), len(nodes)), dtype=dtype, device=device)
    for source, target in graph.edges:
        adjacency[index[target], index[source]] = 1.0
        if undirected:
            adjacency[index[source], index[target]] = 1.0
    if self_loops:
        adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1)
    inv_sqrt = torch.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = degree[nonzero].rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def scaled_cosine_error(prediction: Tensor, target: Tensor, alpha: float = 2.0) -> Tensor:
    """GraphMAE's scaled cosine reconstruction loss."""

    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must be equal-shaped rank-2 tensors, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape[0] == 0:
        raise ValueError("scaled_cosine_error requires at least one target row")
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    return (1.0 - cosine.clamp(-1.0, 1.0)).pow(alpha).mean()


class _GraphConv(nn.Module):
    """Dense normalized-adjacency message passing followed by a learned map."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        return self.linear(adjacency @ features)


class GraphMAEModel(nn.Module):
    """GCN encoder-decoder used by the baseline wrapper."""

    def __init__(self, embedding_dim: int, config: GraphMAEConfig) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.config = config
        type_dim = 2 if config.use_type_features else 0

        encoder_dims = [embedding_dim + type_dim]
        encoder_dims += [config.hidden_dim] * config.encoder_layers
        self.encoder = nn.ModuleList(
            _GraphConv(encoder_dims[i], encoder_dims[i + 1])
            for i in range(config.encoder_layers)
        )
        self.encoder_norms = nn.ModuleList(
            nn.LayerNorm(config.hidden_dim) for _ in range(config.encoder_layers)
        )

        decoder_dims = [config.hidden_dim]
        if config.decoder_layers > 1:
            decoder_dims += [config.hidden_dim] * (config.decoder_layers - 1)
        decoder_dims += [embedding_dim]
        self.decoder = nn.ModuleList(
            _GraphConv(decoder_dims[i], decoder_dims[i + 1])
            for i in range(config.decoder_layers)
        )
        self.decoder_norms = nn.ModuleList(
            nn.LayerNorm(config.hidden_dim) for _ in range(config.decoder_layers - 1)
        )

        self.input_mask_token = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder_mask_token = nn.Parameter(torch.zeros(config.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in [*self.encoder, *self.decoder]:
            nn.init.xavier_uniform_(layer.linear.weight)
            nn.init.zeros_(layer.linear.bias)
        for norm in [*self.encoder_norms, *self.decoder_norms]:
            norm.reset_parameters()
        nn.init.normal_(self.input_mask_token, std=0.02)
        nn.init.normal_(self.decoder_mask_token, std=0.02)

    def forward(
        self,
        semantic_features: Tensor,
        node_types: Tensor,
        adjacency: Tensor,
        input_mask: Tensor,
        remask: Tensor,
    ) -> Tensor:
        """Reconstruct semantic features for every node.

        ``input_mask`` marks unavailable semantic inputs.  ``remask`` marks the
        supervised reconstruction targets whose encoder representations must be
        hidden from the decoder.  Latents are input-masked but are not re-masked
        during observed-node training, so they can serve as structural relays.
        """

        n_nodes = semantic_features.shape[0]
        expected = (n_nodes,)
        if input_mask.shape != expected or remask.shape != expected:
            raise ValueError("input_mask and remask must have shape [n_nodes]")
        masked_semantics = torch.where(
            input_mask[:, None],
            self.input_mask_token[None, :],
            semantic_features,
        )
        if self.config.use_type_features:
            typed = F.one_hot(node_types, num_classes=2).to(masked_semantics.dtype)
            hidden = torch.cat([masked_semantics, typed], dim=-1)
        else:
            hidden = masked_semantics

        for layer, norm in zip(self.encoder, self.encoder_norms):
            hidden = layer(hidden, adjacency)
            hidden = F.gelu(norm(hidden))
            hidden = F.dropout(hidden, p=self.config.dropout, training=self.training)

        hidden = torch.where(
            remask[:, None],
            self.decoder_mask_token[None, :],
            hidden,
        )
        for i, layer in enumerate(self.decoder):
            hidden = layer(hidden, adjacency)
            if i < len(self.decoder) - 1:
                hidden = F.gelu(self.decoder_norms[i](hidden))
                hidden = F.dropout(hidden, p=self.config.dropout, training=self.training)
        return hidden


class GraphMAEBaseline:
    """Leakage-resistant estimator around :class:`GraphMAEModel`.

    Typical use::

        baseline = GraphMAEBaseline(GraphMAEConfig(seed=7)).fit(dev_examples)
        predictions = baseline.infer(test_graph, visible_embeddings)
        baseline.save_checkpoint("outputs/graphmae.pt")

    ``fit`` always reinitializes the model from ``config.seed``.  This makes two
    fits with the same examples, ordering, configuration, and device deterministic.
    """

    def __init__(
        self,
        config: Optional[GraphMAEConfig] = None,
        *,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self.config = config or GraphMAEConfig()
        self.embedding_dim = int(embedding_dim) if embedding_dim is not None else None
        self.device = _resolve_device(self.config.device)
        self.model: Optional[GraphMAEModel] = None
        self.history_: list[float] = []
        self.metadata_: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def _build_model(self, embedding_dim: int) -> None:
        self.embedding_dim = int(embedding_dim)
        self.model = GraphMAEModel(self.embedding_dim, self.config).to(self.device)

    def _validate_embedding_map(
        self,
        graph: Any,
        embeddings: Mapping[str, ArrayLike],
        *,
        require_complete_observed: bool,
        observed_only: bool,
        expected_dim: Optional[int],
    ) -> tuple[dict[str, np.ndarray], int]:
        nodes, observed, _ = _graph_parts(graph)
        node_set = set(nodes)
        keys = set(embeddings)
        if require_complete_observed:
            missing = observed - keys
            if missing:
                raise ValueError(f"training example is missing observed embeddings: {sorted(missing)}")
        allowed = observed if observed_only else node_set
        extra = keys - allowed
        if extra:
            scope = "observed nodes" if observed_only else "graph nodes"
            raise ValueError(f"embedding keys must be {scope}; unexpected: {sorted(extra)}")
        if not keys:
            raise ValueError("embedding mapping must not be empty")

        clean: dict[str, np.ndarray] = {}
        inferred_dim = expected_dim
        for name in nodes:
            if name not in embeddings:
                continue
            value = embeddings[name]
            if isinstance(value, Tensor):
                array = value.detach().cpu().numpy()
            else:
                array = np.asarray(value)
            array = np.asarray(array, dtype=np.float32)
            if array.ndim != 1:
                raise ValueError(f"embedding for {name!r} must have shape [d], got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"embedding for {name!r} contains NaN or infinity")
            if inferred_dim is None:
                inferred_dim = int(array.shape[0])
            if array.shape[0] != inferred_dim:
                raise ValueError(
                    f"embedding dimension mismatch for {name!r}: "
                    f"expected {inferred_dim}, got {array.shape[0]}"
                )
            norm = float(np.linalg.norm(array))
            if self.config.normalize_inputs and norm > 1e-12:
                array = array / norm
            clean[name] = array
        if inferred_dim is None or inferred_dim < 1:
            raise ValueError("could not infer a positive embedding dimension")
        return clean, inferred_dim

    def _prepare_training_example(
        self,
        example: GraphExample,
        embedding_dim: Optional[int],
    ) -> tuple[_PreparedGraph, int]:
        clean, embedding_dim = self._validate_embedding_map(
            example.graph,
            example.observed_embeddings,
            require_complete_observed=True,
            observed_only=True,
            expected_dim=embedding_dim,
        )
        return self._prepare_graph(example.graph, clean, embedding_dim), embedding_dim

    def _prepare_graph(
        self,
        graph: Any,
        known_embeddings: Mapping[str, np.ndarray],
        embedding_dim: int,
    ) -> _PreparedGraph:
        nodes, observed, latents = _graph_parts(graph)
        index = {name: i for i, name in enumerate(nodes)}
        features = torch.zeros(
            (len(nodes), embedding_dim), dtype=torch.float32, device=self.device
        )
        for name, value in known_embeddings.items():
            features[index[name]] = torch.as_tensor(
                value, dtype=torch.float32, device=self.device
            )
        observed_indices = torch.tensor(
            [index[name] for name in nodes if name in observed],
            dtype=torch.long,
            device=self.device,
        )
        latent_mask = torch.tensor(
            [name in latents for name in nodes],
            dtype=torch.bool,
            device=self.device,
        )
        node_types = torch.tensor(
            [0 if name in latents else 1 for name in nodes],
            dtype=torch.long,
            device=self.device,
        )
        adjacency = normalized_adjacency(
            graph,
            undirected=self.config.undirected,
            self_loops=self.config.self_loops,
            device=self.device,
        )
        return _PreparedGraph(
            graph=graph,
            node_names=nodes,
            node_index=index,
            observed_indices=observed_indices,
            latent_mask=latent_mask,
            node_types=node_types,
            features=features,
            adjacency=adjacency,
        )

    @staticmethod
    def _coerce_example(value: Any) -> GraphExample:
        if isinstance(value, GraphExample):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return GraphExample(value[0], value[1])
        raise TypeError(
            "each training item must be GraphExample(graph, observed_embeddings) "
            "or a (graph, observed_embeddings) pair"
        )

    def _sample_observed_mask(
        self,
        observed_indices: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        n_observed = int(observed_indices.numel())
        if n_observed == 0:
            raise ValueError("training graphs must contain at least one observed node")
        n_masked = max(1, int(round(self.config.mask_rate * n_observed)))
        # When possible retain at least one visible semantic input.
        if n_observed > 1:
            n_masked = min(n_masked, n_observed - 1)
        else:
            n_masked = 1
        order = torch.randperm(n_observed, generator=generator)
        selected_cpu = order[:n_masked]
        selected = observed_indices.detach().cpu()[selected_cpu].to(self.device)
        return selected

    def fit(
        self,
        examples: Sequence[Union[GraphExample, tuple[Any, Mapping[str, ArrayLike]]]],
        *,
        progress: Optional[Callable[[int, float], None]] = None,
    ) -> "GraphMAEBaseline":
        """Fit only on the explicitly supplied development examples.

        No path, dataset name, encoder, cache, global split, or held-out graph is
        consulted.  ``progress``, when provided, receives ``(epoch, mean_loss)``.
        """

        raw_examples = [self._coerce_example(value) for value in examples]
        if not raw_examples:
            raise ValueError("fit requires at least one development graph example")

        _seed_everything(self.config.seed, self.config.deterministic)
        prepared: list[_PreparedGraph] = []
        embedding_dim = self.embedding_dim
        for example in raw_examples:
            item, embedding_dim = self._prepare_training_example(example, embedding_dim)
            prepared.append(item)
        assert embedding_dim is not None

        # Fit is a fresh deterministic run even when the wrapper was fitted before.
        self._build_model(embedding_dim)
        assert self.model is not None
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.config.seed)
        self.history_ = []

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            graph_order = torch.randperm(len(prepared), generator=rng).tolist()
            losses: list[float] = []
            for graph_position in graph_order:
                item = prepared[graph_position]
                for _ in range(self.config.masks_per_graph):
                    target_indices = self._sample_observed_mask(item.observed_indices, rng)
                    target_mask = torch.zeros(
                        len(item.node_names), dtype=torch.bool, device=self.device
                    )
                    target_mask[target_indices] = True
                    input_mask = item.latent_mask | target_mask

                    optimizer.zero_grad(set_to_none=True)
                    prediction = self.model(
                        item.features,
                        item.node_types,
                        item.adjacency,
                        input_mask,
                        target_mask,
                    )
                    loss = scaled_cosine_error(
                        prediction[target_mask],
                        item.features[target_mask],
                        alpha=self.config.loss_alpha,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite GraphMAE loss at epoch {epoch}, "
                            f"graph position {graph_position}"
                        )
                    loss.backward()
                    if self.config.grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            mean_loss = float(np.mean(losses))
            self.history_.append(mean_loss)
            if progress is not None:
                progress(epoch, mean_loss)
        self.model.eval()
        return self

    @torch.no_grad()
    def infer(
        self,
        graph: Any,
        visible_embeddings: Mapping[str, ArrayLike],
        *,
        missing_nodes: Optional[Sequence[str]] = None,
    ) -> dict[str, np.ndarray]:
        """Infer embeddings for a fold's missing observed and/or latent nodes.

        The default target set is every graph node absent from
        ``visible_embeddings``.  Callers may pass ``missing_nodes`` to request a
        subset.  Visible latent inputs are accepted for generality, but the standard
        leakage-safe protocol should normally provide visible observed labels only.
        """

        if self.model is None or self.embedding_dim is None:
            raise RuntimeError("GraphMAEBaseline must be fitted or loaded before infer")
        clean, _ = self._validate_embedding_map(
            graph,
            visible_embeddings,
            require_complete_observed=False,
            observed_only=False,
            expected_dim=self.embedding_dim,
        )
        item = self._prepare_graph(graph, clean, self.embedding_dim)
        visible = set(clean)
        if missing_nodes is None:
            requested = [name for name in item.node_names if name not in visible]
        else:
            requested = list(missing_nodes)
            unknown = set(requested) - set(item.node_names)
            if unknown:
                raise ValueError(f"missing_nodes contains names absent from graph: {sorted(unknown)}")
            overlap = set(requested) & visible
            if overlap:
                raise ValueError(
                    "nodes cannot be both visible and requested as missing: "
                    f"{sorted(overlap)}"
                )
        if len(set(requested)) != len(requested):
            raise ValueError("missing_nodes contains duplicates")
        if not requested:
            return {}

        input_mask = torch.tensor(
            [name not in visible for name in item.node_names],
            dtype=torch.bool,
            device=self.device,
        )
        observed = set(graph.observed)
        # Only observed reconstruction targets are re-masked.  A featureless latent
        # keeps its neighbour-conditioned encoder state, since no latent gold target
        # was available during training.
        remask = torch.tensor(
            [name in set(requested) and name in observed for name in item.node_names],
            dtype=torch.bool,
            device=self.device,
        )
        self.model.eval()
        prediction = self.model(
            item.features,
            item.node_types,
            item.adjacency,
            input_mask,
            remask,
        )
        if self.config.normalize_outputs:
            prediction = F.normalize(prediction, dim=-1, eps=1e-8)
        return {
            name: prediction[item.node_index[name]].detach().cpu().numpy().astype(np.float64)
            for name in requested
        }

    predict = infer

    def save_checkpoint(self, checkpoint_path: Union[str, os.PathLike[str]]) -> None:
        """Save weights and configuration, excluding all training examples/data."""

        if self.model is None or self.embedding_dim is None:
            raise RuntimeError("cannot save an unfitted GraphMAEBaseline")
        path = os.fspath(checkpoint_path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        state = {name: value.detach().cpu() for name, value in self.model.state_dict().items()}
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "embedding_dim": self.embedding_dim,
                "config": dataclasses.asdict(self.config),
                "state_dict": state,
                "metadata": dict(self.metadata_),
            },
            path,
        )

    save = save_checkpoint

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, os.PathLike[str]],
        *,
        device: Optional[str] = None,
    ) -> "GraphMAEBaseline":
        """Load a checkpoint onto ``device`` (or its saved/automatic device)."""

        path = os.fspath(checkpoint_path)
        requested_device = device or "auto"
        map_location = _resolve_device(requested_device)
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:  # PyTorch versions predating ``weights_only``.
            payload = torch.load(path, map_location=map_location)
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"not a supported GraphMAE checkpoint: {path}")
        config_values = dict(payload["config"])
        config_values["device"] = requested_device
        config = GraphMAEConfig(**config_values)
        baseline = cls(config, embedding_dim=int(payload["embedding_dim"]))
        baseline._build_model(baseline.embedding_dim)
        assert baseline.model is not None
        baseline.model.load_state_dict(payload["state_dict"], strict=True)
        baseline.model.eval()
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("GraphMAE checkpoint metadata must be a mapping")
        baseline.metadata_ = dict(metadata)
        return baseline

    load = load_checkpoint


def fit_graphmae(
    examples: Sequence[Union[GraphExample, tuple[Any, Mapping[str, ArrayLike]]]],
    config: Optional[GraphMAEConfig] = None,
    *,
    progress: Optional[Callable[[int, float], None]] = None,
) -> GraphMAEBaseline:
    """Functional convenience wrapper around ``GraphMAEBaseline(...).fit``."""

    return GraphMAEBaseline(config).fit(examples, progress=progress)


def _dev_dataset_names(spec: str, excluded: str = "") -> list[str]:
    """Resolve an explicit training list while structurally forbidding held-out data."""
    import pool

    requested = list(pool.DEV) if spec.strip().lower() == "dev" else [
        name.strip() for name in spec.split(",") if name.strip()
    ]
    if not requested:
        raise ValueError("GRAPHMAE_TRAIN_DATASET selected no datasets")
    if len(set(requested)) != len(requested):
        raise ValueError("GRAPHMAE_TRAIN_DATASET contains duplicates")
    outside_dev = set(requested) - set(pool.DEV)
    if outside_dev:
        raise ValueError(
            "GraphMAE training is restricted to pool.DEV; rejected: "
            + ", ".join(sorted(outside_dev))
        )
    excluded_names = {name.strip() for name in excluded.split(",") if name.strip()}
    unknown_exclusions = excluded_names - set(pool.DEV)
    if unknown_exclusions:
        raise ValueError(
            "GRAPHMAE_EXCLUDE contains names outside pool.DEV: "
            + ", ".join(sorted(unknown_exclusions))
        )
    selected = [name for name in requested if name not in excluded_names]
    if not selected:
        raise ValueError("GRAPHMAE_EXCLUDE removed every training dataset")
    return selected


def _install_l3_encoder(device: str, checkpoint: str):
    """Install the same frozen LoRA encoder used by v6/main.py, for CLI training only."""
    import encode
    import lora

    st = lora.load_st(device)
    lora.inject(st)
    lora.load_lora(st, checkpoint)
    st.eval()

    class _LoraST:
        def encode(self, texts, batch_size=1024, normalize_embeddings=True):
            outputs = []
            with torch.no_grad():
                for start in range(0, len(texts), 256):
                    stripped = [
                        text[len("query: "):] if text.startswith("query: ") else text
                        for text in texts[start:start + 256]
                    ]
                    outputs.append(
                        lora.encode_grad(st, stripped, device, max_len=128).cpu().numpy()
                    )
            return np.concatenate(outputs)

    encode._MODEL = _LoraST()
    return encode


def train_graphmae_cli() -> str:
    """Train a frozen GraphMAE-GCN checkpoint strictly from an explicit DEV subset."""
    import pool
    import testbeds

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
    device_spec = os.environ.get(
        "GRAPHMAE_DEVICE", os.environ.get("DEVICE", "auto")
    )
    device = str(_resolve_device(device_spec))
    checkpoint = os.environ.get("GRAPHMAE_CKPT", DEFAULT_CKPT)
    lora_checkpoint = os.environ.get(
        "LORA_CKPT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "l3_lora.pt"),
    )
    if not os.path.isfile(lora_checkpoint):
        raise FileNotFoundError(f"missing L3 LoRA checkpoint: {lora_checkpoint}")

    train_spec = os.environ.get("GRAPHMAE_TRAIN_DATASET", "dev")
    excluded = os.environ.get("GRAPHMAE_EXCLUDE", "")
    names = _dev_dataset_names(train_spec, excluded)
    loaders = {**testbeds.LOADERS, **pool.LOADERS}
    encode = _install_l3_encoder(device, lora_checkpoint)
    examples = []
    for position, name in enumerate(names, 1):
        print(f"[graphmae] encode {position}/{len(names)}: {name}", flush=True)
        dataset = loaders[name]()
        graph = dataset["graph"]
        embeddings = encode.embed([dataset["labels"][node] for node in graph.observed])
        examples.append(
            GraphExample(
                graph,
                {node: embeddings[i] for i, node in enumerate(graph.observed)},
            )
        )

    epochs = int(
        os.environ.get("GRAPHMAE_EPOCHS", os.environ.get("GRAPHMAE_STEPS", 200))
    )
    config = GraphMAEConfig(
        hidden_dim=int(os.environ.get("GRAPHMAE_HID", 128)),
        encoder_layers=int(os.environ.get("GRAPHMAE_LAYERS", 2)),
        decoder_layers=int(os.environ.get("GRAPHMAE_DECODER_LAYERS", 2)),
        dropout=float(os.environ.get("GRAPHMAE_DROPOUT", 0.0)),
        mask_rate=float(os.environ.get("GRAPHMAE_MASK_RATE", 0.5)),
        masks_per_graph=int(os.environ.get("GRAPHMAE_MASKS_PER_GRAPH", 1)),
        learning_rate=float(os.environ.get("GRAPHMAE_LR", 1e-3)),
        weight_decay=float(os.environ.get("GRAPHMAE_WEIGHT_DECAY", 1e-5)),
        epochs=epochs,
        seed=int(os.environ.get("GRAPHMAE_SEED", 0)),
        device=device,
    )
    log_every = max(1, epochs // 20)

    def progress(epoch: int, loss: float) -> None:
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"[graphmae] epoch {epoch:4d}/{epochs}: loss={loss:.6f}", flush=True)

    baseline = GraphMAEBaseline(config).fit(examples, progress=progress)
    baseline.metadata_ = {
        "train_datasets": list(names),
        "excluded_datasets": sorted(
            name.strip() for name in excluded.split(",") if name.strip()
        ),
        "seed": config.seed,
        "encoder": os.environ.get("GRAPHSEM_ENCODER", "e5-large"),
        "lora_checkpoint": os.path.basename(lora_checkpoint),
        "lora_sha256": file_sha256(lora_checkpoint),
        "method": "GraphMAE-GCN causal-graph adaptation",
        "method_version": "graphmae-gcn-causal-imputer-v1",
    }
    baseline.save_checkpoint(checkpoint)
    print(f"[graphmae] saved {checkpoint}", flush=True)
    return checkpoint


__all__ = [
    "DEFAULT_CKPT",
    "GraphExample",
    "GraphMAEConfig",
    "GraphMAEModel",
    "GraphMAEBaseline",
    "fit_graphmae",
    "file_sha256",
    "train_graphmae_cli",
    "normalized_adjacency",
    "scaled_cosine_error",
]


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["train"]:
        train_graphmae_cli()
    else:
        raise SystemExit("usage: python graphmae.py train")
