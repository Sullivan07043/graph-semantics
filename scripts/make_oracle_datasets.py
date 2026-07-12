"""Create deterministic synthetic/oracle Task 1 datasets.

Each dataset contains a known semantic graph, observed-variable labels, and a
linear data-generating process. The files are intentionally simple so the
existing Task 1 pipeline can treat them like ordinary given-graph testbeds.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data"
DEFAULT_SAMPLES = 1200
DEFAULT_SEED = 20260710


def item(item_id, label, parents, loadings, item_type="pure", polarity=None):
    return {
        "id": item_id,
        "label": label,
        "parents": list(parents),
        "loadings": {name: float(value) for name, value in loadings.items()},
        "item_type": item_type,
        "polarity": polarity,
    }


def oracle_clean():
    latents = {
        "L_WORKING_MEMORY": "working memory",
        "L_RELATIONAL_INTEGRATION": "relational integration",
        "L_DIVIDED_ATTENTION": "divided attention",
        "L_TASK_SWITCHING": "task switching",
    }
    groups = {
        "L_WORKING_MEMORY": [
            ("WM_RETAIN", "retain information in working memory"),
            ("WM_SEQUENCE", "remember a sequence of task items"),
            ("WM_MAINTAIN", "maintain task-relevant information"),
            ("WM_SPAN", "working memory span for task material"),
        ],
        "L_RELATIONAL_INTEGRATION": [
            ("RI_PATTERN", "integrate relational patterns"),
            ("RI_NUMERIC", "reason about numerical relations"),
            ("RI_RULES", "combine multiple rule relations"),
            ("RI_FEATURES", "relational reasoning across features"),
        ],
        "L_DIVIDED_ATTENTION": [
            ("DA_SPLIT", "split attention across simultaneous tasks"),
            ("DA_STREAMS", "monitor multiple information streams"),
            ("DA_CROSSMODAL", "coordinate crossmodal attention"),
            ("DA_DUAL", "manage dual-task attention"),
        ],
        "L_TASK_SWITCHING": [
            ("TS_RULES", "switch between task rules"),
            ("TS_RESPONSE", "alternate between response sets"),
            ("TS_SHIFT", "shift a cognitive task set"),
            ("TS_RAPID", "rapid task-rule switching"),
        ],
    }
    observed = [
        item(item_id, label, [latent], {latent: 0.90})
        for latent, values in groups.items()
        for item_id, label in values
    ]
    return {
        "name": "oracle_clean",
        "description": "Dense positive-loading sibling graph for testing ideal recovery.",
        "latents": latents,
        "observed": observed,
        "noise_sd": 0.40,
    }


def oracle_polarity():
    latents = {
        "L_EXTRAVERSION": "extraversion",
        "L_NEUROTICISM": "neuroticism",
    }
    observed = [
        item("E_POS_SOCIAL", "I enjoy lively social interaction.", ["L_EXTRAVERSION"],
             {"L_EXTRAVERSION": 0.90}, polarity="positive"),
        item("E_POS_TALK", "I speak easily with new people.", ["L_EXTRAVERSION"],
             {"L_EXTRAVERSION": 0.85}, polarity="positive"),
        item("E_POS_GATHER", "I seek out social gatherings.", ["L_EXTRAVERSION"],
             {"L_EXTRAVERSION": 0.88}, polarity="positive"),
        item("E_REV_WITHDRAW", "I avoid social interaction.", ["L_EXTRAVERSION"],
             {"L_EXTRAVERSION": -0.90}, polarity="reverse"),
        item("N_POS_WORRY", "I often feel anxious and worried.", ["L_NEUROTICISM"],
             {"L_NEUROTICISM": 0.90}, polarity="positive"),
        item("N_POS_UPSET", "I become upset easily.", ["L_NEUROTICISM"],
             {"L_NEUROTICISM": 0.86}, polarity="positive"),
        item("N_POS_DISTRESS", "I experience frequent emotional distress.", ["L_NEUROTICISM"],
             {"L_NEUROTICISM": 0.88}, polarity="positive"),
        item("N_REV_CALM", "I remain calm under stress.", ["L_NEUROTICISM"],
             {"L_NEUROTICISM": -0.90}, polarity="reverse"),
    ]
    return {
        "name": "oracle_polarity",
        "description": "Positive and reverse-coded items under each latent factor.",
        "latents": latents,
        "observed": observed,
        "noise_sd": 0.38,
    }


def oracle_mixed_parent():
    latents = {
        "L_WORKING_MEMORY": "working memory",
        "L_DIVIDED_ATTENTION": "divided attention",
        "L_RELATIONAL_REASONING": "relational reasoning",
    }
    observed = [
        item("WM_HOLD", "hold information in working memory", ["L_WORKING_MEMORY"],
             {"L_WORKING_MEMORY": 0.90}),
        item("WM_ORDER", "remember the order of task items", ["L_WORKING_MEMORY"],
             {"L_WORKING_MEMORY": 0.86}),
        item("WM_UPDATE", "update information held in memory", ["L_WORKING_MEMORY"],
             {"L_WORKING_MEMORY": 0.84}),
        item("DA_SPLIT", "split attention between concurrent tasks", ["L_DIVIDED_ATTENTION"],
             {"L_DIVIDED_ATTENTION": 0.90}),
        item("DA_MONITOR", "monitor two information streams", ["L_DIVIDED_ATTENTION"],
             {"L_DIVIDED_ATTENTION": 0.86}),
        item("DA_DUAL", "coordinate dual-task attention", ["L_DIVIDED_ATTENTION"],
             {"L_DIVIDED_ATTENTION": 0.84}),
        item("RR_RELATE", "reason about relations between elements", ["L_RELATIONAL_REASONING"],
             {"L_RELATIONAL_REASONING": 0.90}),
        item("RR_INTEGRATE", "integrate multiple relational rules", ["L_RELATIONAL_REASONING"],
             {"L_RELATIONAL_REASONING": 0.86}),
        item("RR_PATTERN", "detect relational patterns", ["L_RELATIONAL_REASONING"],
             {"L_RELATIONAL_REASONING": 0.84}),
        item("MIX_WM_DA", "working memory under divided attention",
             ["L_WORKING_MEMORY", "L_DIVIDED_ATTENTION"],
             {"L_WORKING_MEMORY": 0.64, "L_DIVIDED_ATTENTION": 0.64}, item_type="mixed"),
        item("MIX_WM_RR", "working memory for relational reasoning",
             ["L_WORKING_MEMORY", "L_RELATIONAL_REASONING"],
             {"L_WORKING_MEMORY": 0.64, "L_RELATIONAL_REASONING": 0.64}, item_type="mixed"),
        item("MIX_DA_RR", "relational reasoning under divided attention",
             ["L_DIVIDED_ATTENTION", "L_RELATIONAL_REASONING"],
             {"L_DIVIDED_ATTENTION": 0.64, "L_RELATIONAL_REASONING": 0.64}, item_type="mixed"),
    ]
    return {
        "name": "oracle_mixed_parent",
        "description": "Pure and mixed-parent observed variables for aggregation diagnostics.",
        "latents": latents,
        "observed": observed,
        "noise_sd": 0.40,
    }


def oracle_sparse_sibling():
    latents = {
        "L_WORKING_MEMORY": "working memory",
        "L_RELATIONAL_INTEGRATION": "relational integration",
        "L_DIVIDED_ATTENTION": "divided attention",
        "L_TASK_SWITCHING": "task switching",
    }
    groups = {
        "L_WORKING_MEMORY": [
            ("WM_RETAIN", "retain information in working memory"),
            ("WM_SEQUENCE", "remember a sequence of task items"),
        ],
        "L_RELATIONAL_INTEGRATION": [
            ("RI_PATTERN", "integrate relational patterns"),
            ("RI_NUMERIC", "reason about numerical relations"),
        ],
        "L_DIVIDED_ATTENTION": [
            ("DA_SPLIT", "split attention across simultaneous tasks"),
            ("DA_STREAMS", "monitor multiple information streams"),
        ],
        "L_TASK_SWITCHING": [
            ("TS_RULES", "switch between task rules"),
            ("TS_RESPONSE", "alternate between response sets"),
        ],
    }
    observed = [
        item(item_id, label, [latent], {latent: 0.90})
        for latent, values in groups.items()
        for item_id, label in values
    ]
    return {
        "name": "oracle_sparse_sibling",
        "description": "Two observed items per latent for sibling-density diagnostics.",
        "latents": latents,
        "observed": observed,
        "noise_sd": 0.40,
    }


BUILDERS = {
    "oracle_clean": oracle_clean,
    "oracle_polarity": oracle_polarity,
    "oracle_mixed_parent": oracle_mixed_parent,
    "oracle_sparse_sibling": oracle_sparse_sibling,
}


def sibling_count(observed, target):
    target_parents = set(target["parents"])
    return sum(
        1 for other in observed
        if other["id"] != target["id"] and target_parents & set(other["parents"])
    )


def write_dataset(spec, output_root, n_samples, seed):
    folder = output_root / spec["name"]
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    latent_names = list(spec["latents"])
    latent_scores = {name: rng.normal(size=n_samples) for name in latent_names}
    observed = spec["observed"]
    X = np.empty((n_samples, len(observed)), dtype=np.float64)
    for col, observed_spec in enumerate(observed):
        signal = sum(
            loading * latent_scores[latent]
            for latent, loading in observed_spec["loadings"].items()
        )
        X[:, col] = signal + rng.normal(scale=spec["noise_sd"], size=n_samples)

    with (folder / "data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([entry["id"] for entry in observed])
        writer.writerows(np.round(X, 8))

    with (folder / "codebook.txt").open("w", encoding="utf-8") as handle:
        for entry in observed:
            handle.write(f"{entry['id']}\t{entry['label']}\n")

    dot_lines = ["digraph G {"]
    dot_lines.extend(f"  {latent} [color=red]" for latent in latent_names)
    dot_lines.extend(f"  {entry['id']} [color=blue]" for entry in observed)
    dot_lines.extend(
        f"  {parent} -> {entry['id']}"
        for entry in observed
        for parent in entry["parents"]
    )
    dot_lines.append("}")
    (folder / "graph.dot").write_text("\n".join(dot_lines) + "\n", encoding="utf-8")
    (folder / "latent_labels.json").write_text(
        json.dumps(spec["latents"], indent=2) + "\n", encoding="utf-8"
    )

    metadata = {
        "name": spec["name"],
        "description": spec["description"],
        "generator": {
            "seed": seed,
            "n_samples": n_samples,
            "noise_sd": spec["noise_sd"],
            "process": "linear latent loadings plus independent Gaussian noise",
        },
        "latents": spec["latents"],
        "observed": {
            entry["id"]: {
                **entry,
                "sibling_count": sibling_count(observed, entry),
            }
            for entry in observed
        },
    }
    (folder / "oracle_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--only", choices=sorted(BUILDERS))
    args = parser.parse_args()
    names = [args.only] if args.only else list(BUILDERS)
    for offset, name in enumerate(names):
        folder = write_dataset(BUILDERS[name](), args.output_root, args.n_samples, args.seed + offset)
        print(f"wrote {name}: {folder}")


if __name__ == "__main__":
    main()
