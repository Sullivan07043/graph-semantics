"""Leakage-resistant GraphMAE-GCN adaptation for Task 2 latent readout.

The model is trained only to reconstruct masked *observed* label embeddings on
explicitly supplied development graphs.  Latent nodes never receive a text
embedding or a reconstruction target.  At inference, a latent prediction is the
decoder output of its neighbour-conditioned hidden state.  This is therefore an
indirect GraphMAE latent readout, not an official GraphMAE task and not a model
trained on latent names.

The module has no dataset or text-encoder imports.  That separation is deliberate:
callers must provide the exact training examples and visible fold embeddings.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


CHECKPOINT_FORMAT = "graphmae-gcn-task2-lodo-v1"
METHOD_VERSION = "graphmae-gcn-observed-reconstruction-latent-readout-v1"


def file_sha256(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GraphExample:
    """One training graph containing observed label embeddings only."""

    dataset: str
    graph: Any
    observed_embeddings: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class GraphMAEConfig:
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
    seed: int = 0
    deterministic: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0 < self.mask_rate <= 1:
            raise ValueError("mask_rate must lie in (0, 1]")
        if self.masks_per_graph < 1 or self.epochs < 1:
            raise ValueError("masks_per_graph and epochs must be positive")
        if self.loss_alpha <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid loss or optimizer setting")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")


@dataclass
class _PreparedGraph:
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
        raise RuntimeError(f"CUDA was requested but is unavailable: {spec}")
    return device


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _graph_parts(graph: Any) -> tuple[tuple[str, ...], set[str], set[str]]:
    for attribute in ("nodes", "observed", "latents", "edges"):
        if not hasattr(graph, attribute):
            raise TypeError(f"graph is missing {attribute!r}")
    nodes = tuple(graph.nodes)
    observed = set(graph.observed)
    latents = set(graph.latents)
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("graph nodes must be non-empty and unique")
    if observed & latents or observed | latents != set(nodes):
        raise ValueError("observed and latent nodes must form a disjoint partition")
    for edge in graph.edges:
        if len(edge) != 2 or edge[0] not in set(nodes) or edge[1] not in set(nodes):
            raise ValueError(f"invalid graph edge: {edge!r}")
    return nodes, observed, latents


def normalized_adjacency(graph: Any, device: torch.device) -> Tensor:
    """Symmetric normalized adjacency of the binary undirected projection."""

    nodes, _, _ = _graph_parts(graph)
    index = {node: position for position, node in enumerate(nodes)}
    adjacency = torch.zeros((len(nodes), len(nodes)), dtype=torch.float32, device=device)
    for source, target in graph.edges:
        i, j = index[source], index[target]
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    adjacency.fill_diagonal_(1)
    degree = adjacency.sum(1)
    inverse = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    return inverse[:, None] * adjacency * inverse[None, :]


def scaled_cosine_error(prediction: Tensor, target: Tensor, alpha: float) -> Tensor:
    if prediction.ndim != 2 or prediction.shape != target.shape or prediction.shape[0] == 0:
        raise ValueError("prediction and target must be non-empty equal-shaped matrices")
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    return (1 - cosine.clamp(-1, 1)).pow(alpha).mean()


class _GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        return self.linear(adjacency @ features)


class GraphMAEModel(nn.Module):
    def __init__(self, embedding_dim: int, config: GraphMAEConfig) -> None:
        super().__init__()
        self.config = config
        encoder_dims = [embedding_dim + 2] + [config.hidden_dim] * config.encoder_layers
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
        decoder_remask: Tensor,
    ) -> Tensor:
        semantic = torch.where(
            input_mask[:, None], self.input_mask_token[None, :], semantic_features
        )
        typed = F.one_hot(node_types, num_classes=2).to(semantic.dtype)
        hidden = torch.cat([semantic, typed], dim=-1)
        for layer, norm in zip(self.encoder, self.encoder_norms):
            hidden = F.gelu(norm(layer(hidden, adjacency)))
            hidden = F.dropout(hidden, p=self.config.dropout, training=self.training)
        hidden = torch.where(
            decoder_remask[:, None], self.decoder_mask_token[None, :], hidden
        )
        for position, layer in enumerate(self.decoder):
            hidden = layer(hidden, adjacency)
            if position < len(self.decoder) - 1:
                hidden = F.gelu(self.decoder_norms[position](hidden))
                hidden = F.dropout(hidden, p=self.config.dropout, training=self.training)
        return hidden


class GraphMAEBaseline:
    def __init__(
        self,
        config: Optional[GraphMAEConfig] = None,
        *,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self.config = config or GraphMAEConfig()
        self.device = _resolve_device(self.config.device)
        self.embedding_dim = embedding_dim
        self.model: Optional[GraphMAEModel] = None
        self.history_: list[float] = []
        self.metadata_: dict[str, Any] = {}

    def _clean_embeddings(
        self,
        graph: Any,
        values: Mapping[str, np.ndarray],
        *,
        complete: bool,
        expected_dim: Optional[int],
    ) -> tuple[dict[str, np.ndarray], int]:
        nodes, observed, _ = _graph_parts(graph)
        keys = set(values)
        if keys - observed:
            raise ValueError("semantic inputs may contain observed nodes only")
        if complete and keys != observed:
            raise ValueError("training requires every observed node and no latent node")
        if not keys:
            raise ValueError("at least one visible observed embedding is required")
        cleaned: dict[str, np.ndarray] = {}
        dimension = expected_dim
        for node in nodes:
            if node not in values:
                continue
            vector = np.asarray(values[node], dtype=np.float32)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise ValueError(f"invalid embedding for {node!r}")
            dimension = dimension or int(vector.shape[0])
            if vector.shape != (dimension,):
                raise ValueError(f"embedding dimension mismatch for {node!r}")
            norm = np.linalg.norm(vector)
            cleaned[node] = vector / norm if norm > 1e-12 else vector.copy()
        if dimension is None or dimension < 1:
            raise ValueError("could not infer embedding dimension")
        return cleaned, dimension

    def _prepare(
        self, graph: Any, embeddings: Mapping[str, np.ndarray], embedding_dim: int
    ) -> _PreparedGraph:
        nodes, observed, latents = _graph_parts(graph)
        index = {node: position for position, node in enumerate(nodes)}
        features = torch.zeros((len(nodes), embedding_dim), device=self.device)
        for node, value in embeddings.items():
            features[index[node]] = torch.as_tensor(value, device=self.device)
        return _PreparedGraph(
            node_names=nodes,
            node_index=index,
            observed_indices=torch.tensor(
                [index[node] for node in nodes if node in observed],
                dtype=torch.long,
                device=self.device,
            ),
            latent_mask=torch.tensor(
                [node in latents for node in nodes], dtype=torch.bool, device=self.device
            ),
            node_types=torch.tensor(
                [0 if node in latents else 1 for node in nodes],
                dtype=torch.long,
                device=self.device,
            ),
            features=features,
            adjacency=normalized_adjacency(graph, self.device),
        )

    def _sample_mask(self, observed_indices: Tensor, generator: torch.Generator) -> Tensor:
        count = int(observed_indices.numel())
        masked = max(1, int(round(self.config.mask_rate * count)))
        if count > 1:
            masked = min(masked, count - 1)
        positions = torch.randperm(count, generator=generator)[:masked]
        return observed_indices.detach().cpu()[positions].to(self.device)

    def fit(self, examples: Sequence[GraphExample]) -> "GraphMAEBaseline":
        if not examples:
            raise ValueError("fit requires at least one training graph")
        names = [example.dataset for example in examples]
        if len(names) != len(set(names)):
            raise ValueError("training dataset names must be unique")
        _seed_everything(self.config.seed, self.config.deterministic)
        dimension = self.embedding_dim
        prepared: list[_PreparedGraph] = []
        for example in examples:
            cleaned, dimension = self._clean_embeddings(
                example.graph,
                example.observed_embeddings,
                complete=True,
                expected_dim=dimension,
            )
            prepared.append(self._prepare(example.graph, cleaned, dimension))
        assert dimension is not None
        self.embedding_dim = dimension
        self.model = GraphMAEModel(dimension, self.config).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.history_ = []
        for _epoch in range(self.config.epochs):
            self.model.train()
            losses: list[float] = []
            for graph_position in torch.randperm(len(prepared), generator=generator).tolist():
                item = prepared[graph_position]
                for _ in range(self.config.masks_per_graph):
                    targets = self._sample_mask(item.observed_indices, generator)
                    target_mask = torch.zeros(
                        len(item.node_names), dtype=torch.bool, device=self.device
                    )
                    target_mask[targets] = True
                    optimizer.zero_grad(set_to_none=True)
                    prediction = self.model(
                        item.features,
                        item.node_types,
                        item.adjacency,
                        item.latent_mask | target_mask,
                        target_mask,
                    )
                    loss = scaled_cosine_error(
                        prediction[targets], item.features[targets], self.config.loss_alpha
                    )
                    loss.backward()
                    if self.config.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            self.history_.append(float(np.mean(losses)))
        self.model.eval()
        return self

    @torch.no_grad()
    def infer_latents(
        self, graph: Any, visible_observed_embeddings: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Return one indirect decoder readout for every latent node."""

        if self.model is None or self.embedding_dim is None:
            raise RuntimeError("baseline must be fitted or loaded before inference")
        cleaned, _ = self._clean_embeddings(
            graph,
            visible_observed_embeddings,
            complete=False,
            expected_dim=self.embedding_dim,
        )
        item = self._prepare(graph, cleaned, self.embedding_dim)
        visible = set(cleaned)
        input_mask = torch.tensor(
            [node not in visible for node in item.node_names],
            dtype=torch.bool,
            device=self.device,
        )
        # Historical GraphMAE-GCN behaviour: only supervised observed targets are
        # decoder-remasked.  Latents retain their neighbour-conditioned encoder
        # state because latent text was never a training target.
        decoder_remask = torch.zeros(
            len(item.node_names), dtype=torch.bool, device=self.device
        )
        prediction = F.normalize(
            self.model(
                item.features,
                item.node_types,
                item.adjacency,
                input_mask,
                decoder_remask,
            ),
            dim=-1,
            eps=1e-8,
        )
        _, _, latents = _graph_parts(graph)
        return {
            node: prediction[item.node_index[node]].cpu().numpy().astype(np.float64)
            for node in item.node_names
            if node in latents
        }

    @torch.no_grad()
    def infer_observed_targets(
        self,
        graph: Any,
        visible_observed_embeddings: Mapping[str, np.ndarray],
        target_observed_nodes: Sequence[str],
    ) -> dict[str, np.ndarray]:
        """Reconstruct exactly the hidden observed nodes in a Task 1 fold.

        This is the observed-node readout used by the model's training objective:
        target semantics are replaced by the input mask token and their encoder
        representations are re-masked before decoding.  The narrow contract makes
        leakage checks explicit: every non-target observed node must be supplied,
        no target may be supplied, and latent text is never accepted.
        """

        if self.model is None or self.embedding_dim is None:
            raise RuntimeError("baseline must be fitted or loaded before inference")
        requested = [str(node) for node in target_observed_nodes]
        if not requested:
            raise ValueError("at least one observed target is required")
        if len(requested) != len(set(requested)):
            raise ValueError("target_observed_nodes must be unique")

        nodes, observed, _ = _graph_parts(graph)
        invalid = set(requested) - observed
        if invalid:
            raise ValueError(
                "Task 1 targets must all be observed graph nodes; "
                f"got {sorted(invalid)}"
            )
        cleaned, _ = self._clean_embeddings(
            graph,
            visible_observed_embeddings,
            complete=False,
            expected_dim=self.embedding_dim,
        )
        visible = set(cleaned)
        overlap = visible & set(requested)
        if overlap:
            raise ValueError(
                "observed nodes cannot be both visible and reconstruction targets: "
                f"{sorted(overlap)}"
            )
        expected_targets = observed - visible
        if set(requested) != expected_targets:
            missing = expected_targets - set(requested)
            extra = set(requested) - expected_targets
            raise ValueError(
                "target_observed_nodes must be exactly the hidden observed nodes; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        item = self._prepare(graph, cleaned, self.embedding_dim)
        target_set = set(requested)
        input_mask = torch.tensor(
            [node not in visible for node in nodes],
            dtype=torch.bool,
            device=self.device,
        )
        decoder_remask = torch.tensor(
            [node in target_set for node in nodes],
            dtype=torch.bool,
            device=self.device,
        )
        prediction = F.normalize(
            self.model(
                item.features,
                item.node_types,
                item.adjacency,
                input_mask,
                decoder_remask,
            ),
            dim=-1,
            eps=1e-8,
        )
        return {
            node: prediction[item.node_index[node]].cpu().numpy().astype(np.float64)
            for node in requested
        }

    def save_checkpoint(self, path: os.PathLike[str] | str) -> None:
        if self.model is None or self.embedding_dim is None:
            raise RuntimeError("cannot save an unfitted model")
        target = os.path.abspath(os.fspath(path))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "method_version": METHOD_VERSION,
            "embedding_dim": self.embedding_dim,
            "config": dataclasses.asdict(self.config),
            "state_dict": {
                name: value.detach().cpu() for name, value in self.model.state_dict().items()
            },
            "metadata": dict(self.metadata_),
        }
        descriptor, temporary = tempfile.mkstemp(
            dir=os.path.dirname(target), prefix=".graphmae-", suffix=".tmp"
        )
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @classmethod
    def load_checkpoint(
        cls, path: os.PathLike[str] | str, *, device: str = "auto"
    ) -> "GraphMAEBaseline":
        location = _resolve_device(device)
        try:
            payload = torch.load(path, map_location=location, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=location)
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported GraphMAE checkpoint")
        if payload.get("method_version") != METHOD_VERSION:
            raise ValueError("GraphMAE method-version mismatch")
        config_values = dict(payload["config"])
        config_values["device"] = device
        baseline = cls(
            GraphMAEConfig(**config_values), embedding_dim=int(payload["embedding_dim"])
        )
        baseline.model = GraphMAEModel(baseline.embedding_dim, baseline.config).to(
            baseline.device
        )
        baseline.model.load_state_dict(payload["state_dict"], strict=True)
        baseline.model.eval()
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint lacks provenance metadata")
        baseline.metadata_ = dict(metadata)
        return baseline


__all__ = [
    "CHECKPOINT_FORMAT",
    "METHOD_VERSION",
    "GraphExample",
    "GraphMAEConfig",
    "GraphMAEBaseline",
    "file_sha256",
    "normalized_adjacency",
    "scaled_cosine_error",
]
