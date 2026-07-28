"""TRUNK-4a — nonlinear dependence targets (user order 2026-07-28: no linear approximations
in the training targets; upgrading them BEFORE the retrain, otherwise the nonlinear operator
would be trained to fit linear statistics).

Design: per dataset, ONE joint signal matrix [observed columns | latent scores] and its
GBR-residualized counterpart yield TWO cached distance-correlation matrices. Every constraint
target reads off them:

  |w| edge magnitudes      <- marginal dcor entry (sign kept from the existing signed W)
  bridge floor (obs+latent) <- marginal dcor (replaces Pearson cache + latcon augmented_bridge)
  marginal CI targets       <- sign(Pearson) * marginal dcor, zeroed below the Pearson 2/sqrt(n)
                               noise floor (keep/zero decision stays Pearson-calibrated)
  residual Gram targets     <- sign(residual Pearson) * residual dcor (replaces
                               optimize.partial_residual_corr + latcon augmented_partial_corr)

Residualization: each generated node's signal regressed on its parents' signals with gradient
boosting (sklearn HistGradientBoostingRegressor, default params, fixed seed) — nonparametric,
replacing the linear lstsq. Roots keep their raw signal (as before).

DECLARED linear remnants (THEORY/PLAN): ALS as initializer only; PC1 latent scores as
conditioning summaries; signs from Pearson (dcor is unsigned; polarity is binary).

Caches: outputs/dependence/<name>_nljoint_dcor.npy + <name>_nlres_dcor.npy (+ _names.txt).
Custom graphs (bigfive2) get their own name. Old Pearson caches are never touched.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dependence as dep                                              # noqa: E402

SUB = int(os.environ.get("NLDEP_SUBSAMPLE", 2000))


def _gbr_residual(y, regs):
    from sklearn.ensemble import HistGradientBoostingRegressor
    A = np.stack(regs, 1)
    m = HistGradientBoostingRegressor(random_state=0)
    m.fit(A, y)
    return y - m.predict(A)


def joint_signals(g, X, oi, score):
    obs = list(g.observed)
    lats = [L for L in g.latents if L in score]
    M = np.stack([X[:, oi[o]].astype(float) for o in obs]
                 + [np.asarray(score[L], float) for L in lats], 1)
    return obs + lats, M


def residual_signals(g, X, oi, score, names, M):
    """GBR residual of every generated node on its parents' signals; roots keep raw."""
    idx = {n: i for i, n in enumerate(names)}
    R = M.copy()
    for n in names:
        regs = [np.asarray(score[p], float) for p in g.parents(n)
                if g.is_latent(p) and p in score]
        regs += [X[:, oi[p]].astype(float) for p in g.parents(n) if not g.is_latent(p)]
        if regs:
            R[:, idx[n]] = _gbr_residual(M[:, idx[n]], regs)
    return R


def matrices(g, X, oi, score, name):
    """-> dict(names, idx, marg_dcor, res_dcor, marg_pear, res_pear). dcor matrices cached."""
    names, M = joint_signals(g, X, oi, score)
    rng = np.random.default_rng(0)
    sel = rng.choice(M.shape[0], SUB, replace=False) if M.shape[0] > SUB else slice(None)
    os.makedirs(dep.OUT, exist_ok=True)
    pj = os.path.join(dep.OUT, f"{name}_nljoint_dcor.npy")
    pr = os.path.join(dep.OUT, f"{name}_nlres_dcor.npy")
    pn = os.path.join(dep.OUT, f"{name}_nl_names.txt")
    if os.path.exists(pj) and os.path.exists(pr) and os.path.exists(pn):
        cached = open(pn).read().split("\n")
        assert cached == names, f"{name}: cached nldep names mismatch — delete {pn} to rebuild"
        marg_dcor, res_dcor = np.load(pj), np.load(pr)
        R = residual_signals(g, X, oi, score, names, M)   # cheap enough; needed for signs
    else:
        R = residual_signals(g, X, oi, score, names, M)
        marg_dcor = dep.dcor_mat(M[sel])
        res_dcor = dep.dcor_mat(R[sel])
        np.save(pj, marg_dcor.astype(np.float32))
        np.save(pr, res_dcor.astype(np.float32))
        open(pn, "w").write("\n".join(names))
    marg_pear = np.corrcoef(M.T)
    res_pear = np.corrcoef(R.T)
    np.fill_diagonal(marg_pear, 0.0)
    np.fill_diagonal(res_pear, 0.0)
    return dict(names=names, idx={n: i for i, n in enumerate(names)},
                marg_dcor=np.asarray(marg_dcor, float), res_dcor=np.asarray(res_dcor, float),
                marg_pear=marg_pear, res_pear=res_pear, n_samples=X.shape[0])


# ------------------------------------------------------------------ consumers
def nl_weights(W, mats):
    """|w| <- marginal dcor, sign <- existing signed W (post sign_fix)."""
    idx = mats["idx"]
    out = {}
    for (a, b), w in W.items():
        if a in idx and b in idx and w != 0.0:
            out[(a, b)] = float(np.sign(w) * mats["marg_dcor"][idx[a], idx[b]])
        else:
            out[(a, b)] = float(w)
    return out


def pc_matrix(mats):
    """(names, P): signed residual dcor — the residual Gram anchor targets (obs + latent rows).
    Supersedes optimize.partial_residual_corr + latent_constraints.augmented_partial_corr."""
    P = np.sign(mats["res_pear"]) * mats["res_dcor"]
    np.fill_diagonal(P, 0.0)
    return mats["names"], P


def bridge_dict(mats, lam_upper=0.3, kappa=0.5, q=0.7):
    """Dependence-floor input over observed+latents from marginal dcor.
    Supersedes dependence.load(pearson) + latent_constraints.augmented_bridge."""
    return dict(obs=mats["names"], dep_marg=mats["marg_dcor"],
                lam_upper=lam_upper, kappa=kappa, q=q)


def ci_target(mats, a, b):
    """Marginal CI target: sign(Pearson) * dcor, zero below the Pearson 2/sqrt(n) floor."""
    i, j = mats["idx"][a], mats["idx"][b]
    tau = 2.0 / max(np.sqrt(mats["n_samples"]), 1.0)
    if abs(mats["marg_pear"][i, j]) < tau:
        return 0.0
    return float(np.sign(mats["marg_pear"][i, j]) * mats["marg_dcor"][i, j])


if __name__ == "__main__":                        # warm the caches (lazy on miss otherwise)
    import time
    import pool                                                       # noqa: E402
    import latent_constraints as LC                                   # noqa: E402
    from run_task1 import ALL_LOADERS                                 # noqa: E402
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(pool.DEV) + list(pool.HELDOUT)
    for nm in names:
        ds = ALL_LOADERS[nm]()
        g, X = ds["graph"], ds["X"]
        oi = {o: k for k, o in enumerate(g.observed)}
        W, score = g.estimate_weights(X, oi)
        W, score = LC.sign_fix(g, W, score)
        t0 = time.time()
        matrices(g, X, oi, score, nm)
        print(f"[{time.strftime('%H:%M:%S')}] {nm}: nldep cache ready "
              f"({time.time()-t0:.1f}s)", flush=True)
