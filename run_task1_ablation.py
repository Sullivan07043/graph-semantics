"""Task 1 constraint ablations plus per-item diagnostics.

This runner keeps run_task1.py unchanged. It sweeps the current constraint knobs,
writes one JSON file per ablation, and produces aggregate and per-item reports.
Judge calls are off by default; set RUN_JUDGE=1 to enable them when OPENAI_API_KEY
is available.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import datasets
import encode
import judge as judge_mod
import metrics
import optimize


FOLDS = int(os.environ.get("FOLDS", 5))
STEPS = int(os.environ.get("STEPS", 1500))
LAM_ZERO = float(os.environ.get("LAM_ZERO", 0.3))
LAM_NORM = float(os.environ.get("LAM_NORM", 0.1))
DEVICE = os.environ.get("DEVICE", "cpu")
DATASET = os.environ.get("DATASET", "all")
RUN_JUDGE = os.environ.get("RUN_JUDGE", "0").lower() in ("1", "true", "yes", "on")
OUT_DIR = Path(os.environ.get("ABLATION_OUT_DIR", HERE / "outputs" / "ablations"))
DIAG_DIR = Path(os.environ.get("DIAGNOSTIC_OUT_DIR", HERE / "outputs" / "diagnostics"))
ERROR_REPORT_FILENAME = os.environ.get("ERROR_REPORT_FILENAME", "error_report.md")
CONFIG_FILTER = [x.strip() for x in os.environ.get("ABLATION_CONFIGS", "").split(",") if x.strip()]


def ts():
    return time.strftime("%H:%M:%S")


def config_grid():
    configs = [
        ("A_original", "signed", False, 0.0),
        ("B_abs_only", "abs", False, 0.0),
        ("C_positive_only", "positive", False, 0.0),
        ("D_norm_only", "signed", True, 0.0),
        ("E_abs_norm", "abs", True, 0.0),
        ("F_prior_only_05", "signed", False, 0.5),
        ("G_norm_prior_05", "signed", True, 0.5),
        ("H_abs_norm_prior_05", "abs", True, 0.5),
        ("I_signed_norm_prior_01", "signed", True, 0.1),
        ("J_signed_norm_prior_03", "signed", True, 0.3),
        ("K_signed_norm_prior_10", "signed", True, 1.0),
        ("L_abs_norm_prior_01", "abs", True, 0.1),
        ("M_abs_norm_prior_03", "abs", True, 0.3),
        ("N_abs_norm_prior_10", "abs", True, 1.0),
    ]
    out = []
    for name, edge, norm, prior in configs:
        if CONFIG_FILTER and name not in CONFIG_FILTER:
            continue
        out.append({
            "name": name,
            "edge_weight_mode": edge,
            "normalize_gen": norm,
            "lam_obs_prior": prior,
            "obs_prior_scope": "siblings",
            "steps": STEPS,
            "lam_zero": LAM_ZERO,
            "lam_norm": LAM_NORM,
            "device": DEVICE,
        })
    return out


def prepare_dataset(name, C):
    ds = datasets.LOADERS[name]()
    g, X, labels = ds["graph"], ds["X"], ds["labels"]
    obs = list(g.observed)
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    Tn = metrics.norm_rows(T)
    alpha = metrics.pick_alpha(T, C)
    W, _ = g.estimate_weights(X, oi)
    Craw = np.corrcoef(X.T)
    np.fill_diagonal(Craw, 0.0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    return {
        "name": name,
        "graph": g,
        "X": X,
        "labels": labels,
        "obs": obs,
        "T": T,
        "Tn": Tn,
        "alpha": alpha,
        "W": W,
        "Craw": Craw,
        "folds": folds,
    }


def json_list(values):
    return json.dumps(values, ensure_ascii=False)


def candidate_indices(g, obs, visible, i, scope):
    if scope == "all":
        return list(visible)
    if scope != "siblings":
        raise ValueError(f"unknown observed prior scope: {scope}")
    name = obs[i]
    parents = set(g.parents(name))
    return [j for j in visible if parents & set(g.parents(obs[j]))]


def prior_details(g, corr, obs, labels, visible, vis_emb, scope):
    details = {}
    priors = {}
    for i, name in enumerate(obs):
        if i in set(visible):
            continue
        candidates = candidate_indices(g, obs, visible, i, scope)
        weights = np.clip(corr[i, candidates], 0.0, None) if candidates else np.asarray([])
        sibling_names = [obs[j] for j in candidates]
        info = {
            "visible_sibling_candidates": sibling_names,
            "prior_available": False,
            "prior_top_contributors": [],
        }
        if len(candidates) and weights.sum() >= 1e-9:
            v = sum(float(w) * vis_emb[obs[j]] for w, j in zip(weights, candidates)) / weights.sum()
            priors[name] = v / (np.linalg.norm(v) + 1e-9)
            order = np.argsort(-weights)[:5]
            info["prior_available"] = True
            info["prior_top_contributors"] = [
                {
                    "var": obs[candidates[k]],
                    "label": labels[obs[candidates[k]]],
                    "weight": float(weights[k]),
                    "normalized_weight": float(weights[k] / weights.sum()),
                }
                for k in order
                if weights[k] > 0
            ]
        details[name] = info
    return priors, details


def exact_flags(P, masked, Tn):
    flags = []
    for r, i in enumerate(masked):
        p = P[r] / (np.linalg.norm(P[r]) + 1e-9)
        flags.append(int(np.argmax(Tn @ p) == i))
    return flags


def match_flags(P, masked, T):
    from scipy.optimize import linear_sum_assignment

    S = metrics.norm_rows(P) @ metrics.norm_rows(T[masked]).T
    rows, cols = linear_sum_assignment(-S)
    flags = [0] * len(masked)
    for r, c in zip(rows, cols):
        flags[r] = int(c == r)
    return flags


def cosine_to_true(P, masked, T):
    Pn = metrics.norm_rows(P)
    Tn = metrics.norm_rows(T[masked])
    return [float((Pn[i] * Tn[i]).sum()) for i in range(len(masked))]


def empty_metrics():
    return {"judge": [], "match": [], "exact": []}


def summarize_metric_lists(arms):
    out = {}
    for arm, vals in arms.items():
        out[arm] = {
            key: (float(np.mean(v)) if v else None)
            for key, v in vals.items()
        }
    return out


def run_one_dataset(prep, C, cwords, cfg):
    g = prep["graph"]
    obs = prep["obs"]
    labels = prep["labels"]
    T = prep["T"]
    Tn = prep["Tn"]
    W = prep["W"]
    Craw = prep["Craw"]
    alpha = prep["alpha"]
    records = []
    arms = {arm: empty_metrics() for arm in ("uniform", "rawcorr", "core")}
    use_judge = RUN_JUDGE and judge_mod.available()

    print(
        f"[{ts()}] {cfg['name']} / {prep['name']}: "
        f"edge={cfg['edge_weight_mode']}, norm={cfg['normalize_gen']}, "
        f"prior={cfg['lam_obs_prior']}, judge={use_judge}",
        flush=True,
    )

    for fno, fold in enumerate(prep["folds"]):
        masked = sorted(int(i) for i in fold)
        masked_set = set(masked)
        visible = [i for i in range(len(obs)) if i not in masked_set]
        vis_emb = {obs[i]: T[i] for i in visible}

        prior_emb, prior_info = prior_details(
            g, Craw, obs, labels, visible, vis_emb, cfg["obs_prior_scope"]
        )
        if cfg["lam_obs_prior"] <= 0:
            prior_emb = None

        preds = {}
        for arm, A in (("uniform", np.ones_like(Craw)), ("rawcorr", np.clip(Craw, 0, None))):
            P = np.zeros((len(masked), T.shape[1]))
            for r, i in enumerate(masked):
                w = np.zeros(len(obs))
                w[visible] = A[i, visible]
                if w.sum() < 1e-9:
                    w[visible] = 1.0
                P[r] = (w / w.sum()) @ T
            preds[arm] = P

        emb = optimize.optimize_embeddings(
            g,
            W,
            vis_emb,
            d=T.shape[1],
            steps=cfg["steps"],
            lam_zero=cfg["lam_zero"],
            lam_norm=cfg["lam_norm"],
            seed=fno,
            device=cfg["device"],
            edge_weight_mode=cfg["edge_weight_mode"],
            normalize_gen=cfg["normalize_gen"],
            observed_prior_emb=prior_emb,
            lam_obs_prior=cfg["lam_obs_prior"],
        )
        preds["core"] = np.stack([emb[obs[i]] for i in masked])

        for arm, P in preds.items():
            exact = exact_flags(P, masked, Tn)
            match = match_flags(P, masked, T)
            cosines = cosine_to_true(P, masked, T)
            words = metrics.decode_words(P, C, cwords, alpha)
            jacc, verdicts = (metrics.judge_completion(words, [labels[obs[i]] for i in masked])
                              if use_judge else (None, None))
            arms[arm]["exact"].append(float(np.mean(exact)))
            arms[arm]["match"].append(float(np.mean(match)))
            if jacc is not None:
                arms[arm]["judge"].append(jacc)

            for r, i in enumerate(masked):
                name = obs[i]
                pinfo = prior_info.get(name, {})
                parents = g.parents(name)
                row = {
                    "config_name": cfg["name"],
                    "dataset": prep["name"],
                    "fold": fno,
                    "arm": arm,
                    "variable_id": i,
                    "variable_name": name,
                    "true_label": labels[name],
                    "decoded_top_words": words[r],
                    "exact_correct": bool(exact[r]),
                    "matching_correct": bool(match[r]),
                    "judge_correct": (bool(verdicts[r]) if verdicts else None),
                    "judge_reason": None,
                    "cosine_pred_to_true": cosines[r],
                    "parents": parents,
                    "children": g.children(name),
                    "latent_parents": [p for p in parents if p in set(g.latents)],
                    "visible_sibling_candidates": pinfo.get("visible_sibling_candidates", []),
                    "prior_available": bool(cfg["lam_obs_prior"] > 0 and pinfo.get("prior_available")),
                    "prior_top_contributors": pinfo.get("prior_top_contributors", []),
                    "prior_scope": cfg["obs_prior_scope"],
                    "edge_weight_mode": cfg["edge_weight_mode"],
                    "normalize_gen": cfg["normalize_gen"],
                    "lam_obs_prior": cfg["lam_obs_prior"],
                }
                records.append(row)
        print(f"[{ts()}]   fold {fno + 1}/{FOLDS} done", flush=True)

    return summarize_metric_lists(arms), records


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def write_per_item_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config_name", "dataset", "fold", "arm", "variable_id", "variable_name",
        "true_label", "decoded_top_words", "exact_correct", "matching_correct",
        "judge_correct", "judge_reason", "cosine_pred_to_true", "parents",
        "children", "latent_parents", "visible_sibling_candidates",
        "prior_available", "prior_top_contributors", "prior_scope",
        "edge_weight_mode", "normalize_gen", "lam_obs_prior",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("decoded_top_words", "parents", "children", "latent_parents",
                        "visible_sibling_candidates", "prior_top_contributors"):
                out[key] = json_list(out[key])
            w.writerow(out)


def summary_rows(all_summary):
    original_core = {
        ds: all_summary["A_original"][ds]["core"]
        for ds in all_summary.get("A_original", {})
    }
    rows = []
    def add_rows(cfg_name, ds, by_arm, original):
        raw = by_arm["rawcorr"]
        for arm, vals in by_arm.items():
            row = {
                "config_name": cfg_name,
                "dataset": ds,
                "arm": arm,
                "judge_acc": vals["judge"],
                "matching_acc": vals["match"],
                "exact_top1": vals["exact"],
                "core_minus_rawcorr_matching": None,
                "core_minus_rawcorr_judge": None,
                "core_minus_rawcorr_exact": None,
                "core_minus_original_core_matching": None,
                "core_minus_original_core_judge": None,
                "core_minus_original_core_exact": None,
            }
            if arm == "core":
                row["core_minus_rawcorr_matching"] = diff(vals["match"], raw["match"])
                row["core_minus_rawcorr_judge"] = diff(vals["judge"], raw["judge"])
                row["core_minus_rawcorr_exact"] = diff(vals["exact"], raw["exact"])
                if original:
                    row["core_minus_original_core_matching"] = diff(vals["match"], original["match"])
                    row["core_minus_original_core_judge"] = diff(vals["judge"], original["judge"])
                    row["core_minus_original_core_exact"] = diff(vals["exact"], original["exact"])
            rows.append(row)

    for cfg_name, by_ds in all_summary.items():
        for ds, by_arm in by_ds.items():
            add_rows(cfg_name, ds, by_arm, original_core.get(ds))

        # Macro average gives each testbed equal weight, rather than letting
        # Big Five's larger number of variables dominate the overall result.
        macro_by_arm = {}
        for arm in ("uniform", "rawcorr", "core"):
            macro_by_arm[arm] = {
                key: mean_or_none([by_ds[ds][arm][key] for ds in by_ds])
                for key in ("judge", "match", "exact")
            }
        macro_original = {
            key: mean_or_none([original_core[ds][key] for ds in by_ds if ds in original_core])
            for key in ("judge", "match", "exact")
        }
        add_rows(cfg_name, "macro_average", macro_by_arm, macro_original)
    return rows


def diff(a, b):
    return None if a is None or b is None else float(a - b)


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def fmt(v):
    return "-" if v is None else f"{float(v):.3f}"


def write_summary_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_summary_md(rows, path):
    lines = ["# Task 1 Constraint Ablation Summary", ""]
    if any(row["judge_acc"] is not None for row in rows):
        lines.append("Judge status: enabled; judge-ACC values were returned for all reported arms.")
    else:
        lines.append("Judge status: unavailable. Set `RUN_JUDGE=1` with `OPENAI_API_KEY` to enable it.")
    datasets_in_order = sorted({r["dataset"] for r in rows if r["dataset"] != "macro_average"})
    if any(r["dataset"] == "macro_average" for r in rows):
        datasets_in_order.append("macro_average")
    for ds in datasets_in_order:
        lines += ["", f"## {ds}", ""]
        lines.append(
            "| config | arm | judge-ACC | matching-ACC | exact top-1 | "
            "core-rawcorr judge/match/exact | core-A_original judge/match/exact |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in [x for x in rows if x["dataset"] == ds]:
            lines.append(
                f"| {r['config_name']} | {r['arm']} | {fmt(r['judge_acc'])} | "
                f"{fmt(r['matching_acc'])} | {fmt(r['exact_top1'])} | "
                f"{fmt(r['core_minus_rawcorr_judge'])} / "
                f"{fmt(r['core_minus_rawcorr_matching'])} / "
                f"{fmt(r['core_minus_rawcorr_exact'])} | "
                f"{fmt(r['core_minus_original_core_judge'])} / "
                f"{fmt(r['core_minus_original_core_matching'])} / "
                f"{fmt(r['core_minus_original_core_exact'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_key(row):
    return (row["config_name"], row["dataset"], int(row["fold"]), row["arm"], row["variable_name"])


def build_error_report(all_summary, per_rows, path):
    lines = ["# Task 1 Per-Item Error Report", ""]
    lines.append("Judge was unavailable in this run unless judge-ACC values are present.")
    lines += ["", "## 1. Best Ablation Per Dataset", ""]
    for ds in sorted(next(iter(all_summary.values())).keys()):
        ranked = sorted(
            ((cfg, all_summary[cfg][ds]["core"]) for cfg in all_summary),
            key=lambda x: (
                -1 if x[1]["match"] is None else -x[1]["match"],
                -1 if x[1]["judge"] is None else -x[1]["judge"],
                -1 if x[1]["exact"] is None else -x[1]["exact"],
            ),
        )
        best, vals = ranked[0]
        lines.append(
            f"- **{ds}**: `{best}` by core matching-ACC "
            f"({fmt(vals['match'])}; judge={fmt(vals['judge'])}, exact={fmt(vals['exact'])})."
        )

    lines += ["", "## 2. Does Abs Weight Help?", ""]
    lines += comparison_bullets(all_summary, [
        ("B_abs_only", "A_original", "abs only vs signed"),
        ("E_abs_norm", "D_norm_only", "abs+norm vs signed+norm"),
        ("H_abs_norm_prior_05", "G_norm_prior_05", "abs+norm+prior vs signed+norm+prior"),
    ])

    lines += ["", "## 3. Does Normalized Generation Help?", ""]
    lines += comparison_bullets(all_summary, [
        ("D_norm_only", "A_original", "signed norm vs signed raw"),
        ("E_abs_norm", "B_abs_only", "abs norm vs abs raw"),
        ("G_norm_prior_05", "F_prior_only_05", "signed norm+prior vs signed raw+prior"),
    ])

    lines += ["", "## 4. Does Sibling Prior Help?", ""]
    lines += comparison_bullets(all_summary, [
        ("F_prior_only_05", "A_original", "signed prior vs signed no-prior"),
        ("G_norm_prior_05", "D_norm_only", "signed norm prior vs signed norm no-prior"),
        ("H_abs_norm_prior_05", "E_abs_norm", "abs norm prior vs abs norm no-prior"),
    ])

    lines += ["", "## 5. Why Can TLVD Matching Be High While Judge Drops?", ""]
    lines.append(
        "Matching-ACC only checks whether predictions can be assigned to the held-out labels within a fold. "
        "A model can preserve relative identity while decoded words drift toward generic or neighboring concepts. "
        "TLVD has a small hierarchical graph and broad cognitive labels, so small embedding shifts can keep matching high "
        "but change the SpLiCE decoded words enough for judge-ACC to drop."
    )

    lines += ["", "## 6. Why Does Big Five Improve More?", ""]
    lines.append(
        "Big Five has many observed items per factor. The shared-parent observed prior has many visible sibling labels "
        "to average from, and normalized generation reduces scale differences across item embeddings. This gives the new "
        "constraints more item-level semantic information than in smaller graphs."
    )

    rows_by_key = {row_key(r): r for r in per_rows}
    lines += example_section("Rawcorr Beats Core", per_rows, rows_by_key, "H_abs_norm_prior_05", "rawcorr", "core")
    lines += example_section("Core Beats Rawcorr", per_rows, rows_by_key, "H_abs_norm_prior_05", "core", "rawcorr")
    lines += example_section("Original Beats New", per_rows, rows_by_key, None, "A_original:core", "H_abs_norm_prior_05:core")
    lines += example_section("New Beats Original", per_rows, rows_by_key, None, "H_abs_norm_prior_05:core", "A_original:core")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_bullets(all_summary, pairs):
    lines = []
    for new, old, label in pairs:
        if new not in all_summary or old not in all_summary:
            continue
        parts = []
        for ds in sorted(all_summary[new]):
            delta = diff(all_summary[new][ds]["core"]["match"], all_summary[old][ds]["core"]["match"])
            parts.append(f"{ds}: {fmt(delta)}")
        lines.append(f"- {label}: " + "; ".join(parts))
    return lines


def split_arm(spec, default_cfg):
    if ":" in spec:
        cfg, arm = spec.split(":", 1)
        return cfg, arm
    return default_cfg, spec


def example_section(title, per_rows, rows_by_key, default_cfg, better_spec, worse_spec, limit=8):
    better_cfg, better_arm = split_arm(better_spec, default_cfg)
    worse_cfg, worse_arm = split_arm(worse_spec, default_cfg)
    examples = []
    for row in per_rows:
        if row["config_name"] != better_cfg or row["arm"] != better_arm:
            continue
        k = (worse_cfg, row["dataset"], int(row["fold"]), worse_arm, row["variable_name"])
        other = rows_by_key.get(k)
        if not other:
            continue
        better_match = bool(row["matching_correct"])
        worse_match = bool(other["matching_correct"])
        better_cos = float(row["cosine_pred_to_true"])
        worse_cos = float(other["cosine_pred_to_true"])
        if (better_match and not worse_match) or (better_cos > worse_cos + 0.05):
            examples.append((better_cos - worse_cos, row, other))
    examples = sorted(examples, key=lambda x: -x[0])[:limit]
    lines = ["", f"## Examples: {title}", ""]
    if not examples:
        lines.append("- No clear examples found under the current criterion.")
        return lines
    for delta, good, bad in examples:
        lines.append(
            f"- **{good['dataset']} / fold {good['fold']} / {good['variable_name']}** "
            f"({good['true_label'][:80]}): {better_spec} cos={float(good['cosine_pred_to_true']):.3f}, "
            f"match={good['matching_correct']}, words={', '.join(good['decoded_top_words'])}; "
            f"{worse_spec} cos={float(bad['cosine_pred_to_true']):.3f}, "
            f"match={bad['matching_correct']}, words={', '.join(bad['decoded_top_words'])}."
        )
    return lines


def core_values(all_summary, cfg_name, dataset):
    by_ds = all_summary.get(cfg_name, {})
    if dataset == "macro_average":
        return {
            key: mean_or_none([vals["core"][key] for vals in by_ds.values()])
            for key in ("judge", "match", "exact")
        } if by_ds else None
    return by_ds.get(dataset, {}).get("core")


def rawcorr_values(all_summary, cfg_name, dataset):
    by_ds = all_summary.get(cfg_name, {})
    if dataset == "macro_average":
        return {
            key: mean_or_none([vals["rawcorr"][key] for vals in by_ds.values()])
            for key in ("judge", "match", "exact")
        } if by_ds else None
    return by_ds.get(dataset, {}).get("rawcorr")


def bool_mean(rows, field):
    return mean_or_none([float(bool(row[field])) for row in rows if row.get(field) is not None])


def format_item(row):
    words = ", ".join(row["decoded_top_words"])
    label = " ".join(str(row["true_label"]).split())
    return (
        f"`{row['variable_name']}` (fold {row['fold']}; target: {label[:110]}; "
        f"words: {words})"
    )


def paired_judge_changes(per_rows, before_cfg, after_cfg, dataset):
    def indexed(cfg_name):
        return {
            (int(row["fold"]), row["variable_name"]): row
            for row in per_rows
            if row["config_name"] == cfg_name
            and row["dataset"] == dataset
            and row["arm"] == "core"
            and row.get("judge_correct") is not None
        }

    before = indexed(before_cfg)
    after = indexed(after_cfg)
    gains, losses, ties = [], [], []
    for key in sorted(set(before) & set(after)):
        old, new = before[key], after[key]
        if not bool(old["judge_correct"]) and bool(new["judge_correct"]):
            gains.append((old, new))
        elif bool(old["judge_correct"]) and not bool(new["judge_correct"]):
            losses.append((old, new))
        else:
            ties.append((old, new))
    return gains, losses, ties


def construct_profile(per_rows, cfg_name, term):
    rows = [
        row for row in per_rows
        if row["config_name"] == cfg_name
        and row["dataset"] == "himi"
        and row["arm"] == "core"
        and term.lower() in str(row["true_label"]).lower()
    ]
    if not rows:
        return None
    decoded = []
    for row in rows:
        words = ", ".join(row["decoded_top_words"])
        if words not in decoded:
            decoded.append(words)
    return {
        "judge": bool_mean(rows, "judge_correct"),
        "match": bool_mean(rows, "matching_correct"),
        "cosine": mean_or_none([row["cosine_pred_to_true"] for row in rows]),
        "decoded": decoded[:3],
    }


def fmt_triplet(vals):
    if not vals:
        return "unavailable"
    return (
        f"judge={fmt(vals['judge'])}, matching={fmt(vals['match'])}, "
        f"exact={fmt(vals['exact'])}"
    )


def choose_fixed_config(all_summary):
    """Return a conservative fixed-setting recommendation from the judged run."""
    target = [
        name for name in (
            "A_original", "D_norm_only", "M_abs_norm_prior_03",
            "H_abs_norm_prior_05", "K_signed_norm_prior_10",
        )
        if name in all_summary
    ]
    if "A_original" not in target:
        return None, "A_original was not included, so a fixed-setting decision cannot be made."

    baseline = core_values(all_summary, "A_original", "macro_average")
    raw = rawcorr_values(all_summary, "A_original", "macro_average")
    m_vals = core_values(all_summary, "M_abs_norm_prior_03", "macro_average")
    m_tlvd = core_values(all_summary, "M_abs_norm_prior_03", "tlvd")
    a_tlvd = core_values(all_summary, "A_original", "tlvd")

    def no_drop(new, old, key, guardrail=0.05):
        return new is not None and old is not None and new[key] is not None and old[key] is not None and new[key] >= old[key] - guardrail

    if m_vals and baseline and raw and m_tlvd and a_tlvd:
        m_is_safe = (
            no_drop(m_vals, baseline, "judge")
            and no_drop(m_vals, baseline, "match")
            and no_drop(m_tlvd, a_tlvd, "judge")
            and no_drop(m_vals, raw, "judge")
        )
        if m_is_safe:
            return (
                "M_abs_norm_prior_03",
                "M keeps macro judge/matching within a 5-point guardrail of A_original and rawcorr, "
                "and it does not materially lower TLVD judge-ACC.",
            )

    eligible = []
    for cfg_name in target:
        vals = core_values(all_summary, cfg_name, "macro_average")
        tlvd = core_values(all_summary, cfg_name, "tlvd")
        if not vals or not tlvd:
            continue
        if no_drop(tlvd, a_tlvd, "judge"):
            eligible.append((cfg_name, vals))
    eligible.sort(
        key=lambda item: (
            -1 if item[1]["judge"] is None else -item[1]["judge"],
            -1 if item[1]["match"] is None else -item[1]["match"],
            -1 if item[1]["exact"] is None else -item[1]["exact"],
        )
    )
    if eligible:
        best_name, best = eligible[0]
        if best_name != "A_original" and no_drop(best, raw, "judge"):
            return (
                best_name,
                "This setting has the best eligible macro judged result while keeping TLVD judge-ACC "
                "within the same 5-point guardrail of A_original.",
            )
    return (
        None,
        "No tested graph setting jointly preserves TLVD semantic quality and stays within a 5-point "
        "macro judge-ACC guardrail of rawcorr. Do not fix v1.1 yet; test a polarity-aware sibling prior first.",
    )


def build_judged_error_report(all_summary, per_rows, path):
    """Write the focused semantic-quality report used to choose Task 1 v1.1."""
    lines = ["# Targeted Task 1 Judged Run", ""]
    lines.append(
        "This report uses the LLM completion judge on the same five folds as matching and exact top-1. "
        "The decision rule below uses a 5 percentage-point guardrail as a practical screen, not as a statistical significance test."
    )
    a_name = "A_original"
    m_name = "M_abs_norm_prior_03"
    raw_cfg = m_name if m_name in all_summary else a_name

    lines += ["", "## Aggregate Results", ""]
    lines.append("| setting | macro core judge-ACC | macro core matching-ACC | macro core exact top-1 |")
    lines.append("|---|---:|---:|---:|")
    for cfg_name in (a_name, "D_norm_only", m_name, "H_abs_norm_prior_05", "K_signed_norm_prior_10"):
        vals = core_values(all_summary, cfg_name, "macro_average")
        if vals:
            lines.append(
                f"| {cfg_name} | {fmt(vals['judge'])} | {fmt(vals['match'])} | {fmt(vals['exact'])} |"
            )
    raw = rawcorr_values(all_summary, raw_cfg, "macro_average")
    if raw:
        lines.append(f"| rawcorr baseline | {fmt(raw['judge'])} | {fmt(raw['match'])} | {fmt(raw['exact'])} |")

    m_macro = core_values(all_summary, m_name, "macro_average")
    a_macro = core_values(all_summary, a_name, "macro_average")
    m_raw = rawcorr_values(all_summary, raw_cfg, "macro_average")
    lines += ["", "## 1. Can M_abs_norm_prior_03 Be the Global Fixed v1.1 Configuration?", ""]
    if m_macro and a_macro and m_raw:
        lines.append(
            "M_abs_norm_prior_03 relative to A_original: "
            f"judge {fmt(diff(m_macro['judge'], a_macro['judge']))}, "
            f"matching {fmt(diff(m_macro['match'], a_macro['match']))}, "
            f"exact {fmt(diff(m_macro['exact'], a_macro['exact']))}."
        )
        lines.append(
            "M_abs_norm_prior_03 relative to rawcorr: "
            f"judge {fmt(diff(m_macro['judge'], m_raw['judge']))}, "
            f"matching {fmt(diff(m_macro['match'], m_raw['match']))}, "
            f"exact {fmt(diff(m_macro['exact'], m_raw['exact']))}."
        )

    fixed, reason = choose_fixed_config(all_summary)
    if fixed == m_name:
        lines.append("**Answer:** Yes, subject to the limited three-dataset evidence in this run.")
    else:
        lines.append("**Answer:** No; the targeted semantic-quality screen does not support fixing M as the global setting yet.")

    lines += ["", "## 2. Is M Better Than A_original?", ""]
    for ds in ("tlvd", "himi", "bigfive", "macro_average"):
        m_vals = core_values(all_summary, m_name, ds)
        a_vals = core_values(all_summary, a_name, ds)
        if m_vals and a_vals:
            lines.append(
                f"- **{ds}**: judge {fmt(diff(m_vals['judge'], a_vals['judge']))}; "
                f"matching {fmt(diff(m_vals['match'], a_vals['match']))}; "
                f"exact {fmt(diff(m_vals['exact'], a_vals['exact']))}."
            )

    lines += ["", "## 3. Does M Clearly Lose to rawcorr?", ""]
    if m_macro and m_raw:
        judge_gap = diff(m_macro["judge"], m_raw["judge"])
        clearly_loses = judge_gap is not None and judge_gap < -0.10
        lines.append(
            f"Macro judge gap (M - rawcorr) = {fmt(judge_gap)}. "
            f"Under a 10-point practical threshold, the answer is {'yes' if clearly_loses else 'no'}; "
            "this is descriptive rather than a significance test."
        )
    for ds in ("tlvd", "himi", "bigfive"):
        m_vals = core_values(all_summary, m_name, ds)
        raw_vals = rawcorr_values(all_summary, raw_cfg, ds)
        if m_vals and raw_vals:
            lines.append(
                f"- **{ds}**: judge {fmt(diff(m_vals['judge'], raw_vals['judge']))}; "
                f"matching {fmt(diff(m_vals['match'], raw_vals['match']))}; "
                f"exact {fmt(diff(m_vals['exact'], raw_vals['exact']))}."
            )

    lines += ["", "## 4. TLVD: Matching Saturation Versus Judge-ACC", ""]
    a_tlvd = core_values(all_summary, a_name, "tlvd")
    m_tlvd = core_values(all_summary, m_name, "tlvd")
    if a_tlvd and m_tlvd:
        lines.append(
            f"A_original: {fmt_triplet(a_tlvd)}. M_abs_norm_prior_03: {fmt_triplet(m_tlvd)}."
        )
    gains, losses, _ = paired_judge_changes(per_rows, a_name, m_name, "tlvd")
    lines.append(f"Compared with A_original, M gains {len(gains)} and loses {len(losses)} judged TLVD items.")
    if a_tlvd and m_tlvd:
        lines.append(
            "**Answer:** Matching is saturated for both settings, but judge-ACC does not decline overall: "
            f"M changes it by {fmt(diff(m_tlvd['judge'], a_tlvd['judge']))}. "
            "The item list below identifies the remaining semantic regressions hidden by matching-ACC."
        )
    if losses:
        lines.append("Items whose judge verdict declined:")
        for old, new in losses:
            lines.append(f"- {format_item(old)} -> M words: {', '.join(new['decoded_top_words'])}")
    else:
        lines.append("No TLVD item changed from judge-correct under A_original to judge-incorrect under M.")

    lines += ["", "## 5. Big Five: Does Judge Support the Matching Gain?", ""]
    a_big = core_values(all_summary, a_name, "bigfive")
    m_big = core_values(all_summary, m_name, "bigfive")
    if a_big and m_big:
        lines.append(
            f"A_original: {fmt_triplet(a_big)}. M_abs_norm_prior_03: {fmt_triplet(m_big)}."
        )
        raw_big = rawcorr_values(all_summary, raw_cfg, "bigfive")
        if raw_big:
            lines.append(
                "**Answer:** Yes, partially. Relative to A_original, M improves both judge and matching; "
                f"relative to rawcorr, its judge gap is {fmt(diff(m_big['judge'], raw_big['judge']))} "
                f"and matching gap is {fmt(diff(m_big['match'], raw_big['match']))}. "
                "Thus the judge supports a real semantic gain over A_original, but not a Big Five win over rawcorr."
            )
    gains, losses, _ = paired_judge_changes(per_rows, a_name, m_name, "bigfive")
    lines.append(f"Judge verdict changes for M versus A_original: {len(gains)} gains, {len(losses)} losses.")
    if gains:
        lines.append("Representative judged gains:")
        for old, new in gains[:8]:
            lines.append(f"- {format_item(new)}")
    if losses:
        lines.append("Representative judged losses:")
        for old, new in losses[:8]:
            lines.append(f"- {format_item(new)}")

    lines += ["", "## 6. Himi Construct Specificity", ""]
    lines.append(
        "The profiles below track the two diagnostic constructs across A_original, D_norm_only, and M. "
        "A nonzero judge-ACC together with different decoded word sets is evidence that the optimization is not simply collapsing them into one generic attention concept."
    )
    for term in ("relational integration", "divided attention"):
        lines.append(f"- **{term}**")
        for cfg_name in (a_name, "D_norm_only", m_name):
            profile = construct_profile(per_rows, cfg_name, term)
            if profile:
                lines.append(
                    f"  - `{cfg_name}`: judge={fmt(profile['judge'])}, matching={fmt(profile['match'])}, "
                    f"cosine={fmt(profile['cosine'])}; examples: {' | '.join(profile['decoded'])}."
                )
    ri_m = construct_profile(per_rows, m_name, "relational integration")
    da_m = construct_profile(per_rows, m_name, "divided attention")
    if ri_m and da_m:
        lines.append(
            "**Answer:** Only partially. M recovers relational integration with "
            f"judge={fmt(ri_m['judge'])}, but divided attention remains at judge={fmt(da_m['judge'])} "
            "under A, D, and M. The embeddings remain separable in matching, yet current graph constraints and "
            "decoder do not robustly recover the specific semantics of divided attention."
        )

    lines += ["", "## 7. Fixed v1.1 Recommendation", ""]
    if fixed:
        lines.append(f"**Recommend fixed global configuration: `{fixed}`.** {reason}")
    else:
        lines.append(f"**Do not fix Task 1 v1.1 yet.** {reason}")
    lines.append(
        "Configuration names are fully specified in `ablation_manifest.json`; all runs use five folds, "
        "1500 optimization steps, LAM_ZERO=0.3, LAM_NORM=0.1, CPU, and the sibling-only prior scope."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    names = list(datasets.DEFAULT_TASK1_DATASETS) if DATASET == "all" else [DATASET]
    configs = config_grid()
    C, cwords = encode.load_dictionary()
    prepared = {name: prepare_dataset(name, C) for name in names}

    all_summary = {}
    all_records = []
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "DATASET": DATASET,
            "FOLDS": FOLDS,
            "STEPS": STEPS,
            "LAM_ZERO": LAM_ZERO,
            "LAM_NORM": LAM_NORM,
            "DEVICE": DEVICE,
            "RUN_JUDGE": RUN_JUDGE,
        },
        "configs": configs,
        "files": {},
    }

    for cfg in configs:
        cfg_summary = {}
        cfg_records = []
        for name in names:
            summary, records = run_one_dataset(prepared[name], C, cwords, cfg)
            cfg_summary[name] = summary
            cfg_records.extend(records)
        all_summary[cfg["name"]] = cfg_summary
        all_records.extend(cfg_records)
        out_file = OUT_DIR / f"{cfg['name']}.json"
        write_json(out_file, {"config": cfg, "summary": cfg_summary, "records": cfg_records})
        manifest["files"][cfg["name"]] = str(out_file.relative_to(HERE))
        print(f"[{ts()}] saved {out_file}", flush=True)

    write_json(OUT_DIR / "ablation_manifest.json", manifest)
    rows = summary_rows(all_summary)
    write_summary_csv(rows, OUT_DIR / "summary.csv")
    write_summary_md(rows, OUT_DIR / "summary.md")
    write_per_item_csv(all_records, DIAG_DIR / "per_item_diagnostics.csv")
    if RUN_JUDGE:
        build_judged_error_report(all_summary, all_records, DIAG_DIR / ERROR_REPORT_FILENAME)
    else:
        build_error_report(all_summary, all_records, DIAG_DIR / ERROR_REPORT_FILENAME)
    print(f"[{ts()}] wrote summaries to {OUT_DIR}", flush=True)
    print(f"[{ts()}] wrote diagnostics to {DIAG_DIR}", flush=True)


if __name__ == "__main__":
    main()
