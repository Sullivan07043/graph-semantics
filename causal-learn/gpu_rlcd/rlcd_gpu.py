"""GPU-batched RLCD without touching the vendored implementation.

The vendored RLCD stays byte-identical to upstream. This module provides RLCD_gpu(), which
patches two module attributes of RLCD_alg for the duration of one call and restores them:

  findClusters_at_k_by_nonsinks -> a copy of the upstream function whose only change is the
      test schedule: surviving subsets are buffered, their canonical correlations computed in
      one batched GPU call per chunk (ranktest.prime), then the upstream per-subset logic runs
      unchanged. Same tests, same decisions, different schedule.
  findClusters_at_k_mp -> a serial loop with the exact aggregation semantics of the upstream
      multiprocessing version (collect res_for_add across all nonsinks groups, then add).
      A GPU-resident rank test cannot cross loky process boundaries.

Certification: certify_gpu_ranktest.py must PASS (decision parity + identical end-to-end
graph vs the untouched CPU path) before results from RLCD_gpu are used.

Usage:
    from gpu_ranktest import GpuRankTest
    from rlcd_gpu import RLCD_gpu
    cg = RLCD_gpu(X, ranktest_method=GpuRankTest(X), node_names=names)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "upstream", "causal-learn"))

import numpy as np

import causallearn.search.HiddenCausal.RLCD.RLCD_alg as base
from causallearn.search.HiddenCausal.RLCD.DSU import DSU

M = base.M
LOGGER = base.LOGGER
setLength = base.setLength
setDifference = base.setDifference
getVarNames = base.getVarNames

BATCH_CHUNK = int(os.environ.get("RLCD_BATCH_CHUNK", 100000))


def _rank_cols(xvars, G, As, Bs, nonLeafs):
    """Column indices structuralRankTest will use, mirrored from its first lines."""
    Ameasures = getVarNames(G.pickAllMeasures(As)) + nonLeafs
    Bmeasures = getVarNames(G.pickAllMeasures(Bs)) + nonLeafs
    Ameasures = list(set(Ameasures))
    Bmeasures = list(set(Bmeasures))
    pcols = [xvars.index(a) for a in Ameasures]
    qcols = [xvars.index(b) for b in Bmeasures]
    return pcols, qcols


def _by_nonsinks_batched(G, k, nonsinks, parameters):
    """Upstream findClusters_at_k_by_nonsinks, test schedule batched. Pruning, test and
    v-structure logic are copied verbatim; only the buffering around the test is new."""
    terminate = False
    found = False
    res_for_add = []

    num_nonsinks = len(nonsinks)

    current_activeSet = G.activeSet.copy()
    current_ChildrenOfNonAtomicsSet = G.ChildrenOfNonAtomicsSet.copy()

    for temp in nonsinks:
        current_activeSet.discard(G.X_dict[temp])
        current_ChildrenOfNonAtomicsSet.discard(G.X_dict[temp])

    if k - num_nonsinks > setLength(current_activeSet) / 2 - 1:
        terminate = True
        return (found, terminate, res_for_add)

    if k != len(nonsinks):
        for temp in G.all_nb_set:
            if temp in G.X_dict and G.X_dict[temp] in current_activeSet:
                current_activeSet.discard(G.X_dict[temp])

    allSubsets = [x for x in M.generateSubsetMinimal(current_activeSet, k - num_nonsinks)]

    for v in current_activeSet:
        if len(v) >= k - num_nonsinks + 1 and k - num_nonsinks != 0:
            tempset = current_activeSet.copy()
            tempset.remove(v)
            additionalls = [x for x in M.generateSubsetMinimal(tempset, 0)]
            for x in additionalls:
                x.add(v)
            allSubsets = allSubsets + additionalls

    if allSubsets == [set()]:
        terminate = True
        return (found, terminate, res_for_add)

    ranktest = parameters['ranktest_method']
    batching = hasattr(ranktest, 'prime')
    pending = []

    def process_survivor(As, Bs):
        nonlocal found
        fail_to_reject, rk = base.structuralRankTest(
            parameters['xvars'], parameters['ranktest_method'], parameters['alpha_dict'],
            G, As, Bs, k, list(nonsinks))

        if fail_to_reject:
            LOGGER.info(f"   {As} is rank deficient! given {nonsinks}, Bs:{Bs}")
            v_structure_found = False
            if parameters['check_v']:
                for num_colider in range(1, k - num_nonsinks + 1):
                    num_subAs = k - num_nonsinks + 1 - num_colider
                    for subAs in M.generateSubsetMinimal(As, num_subAs - 1):
                        test_subAs, rk_subAs = base.structuralRankTest(
                            parameters['xvars'], parameters['ranktest_method'],
                            parameters['alpha_dict'], G, subAs, Bs,
                            num_nonsinks + num_subAs - 1, list(nonsinks))
                        if test_subAs:
                            LOGGER.info(f"   {As} has v structure! subAs:{subAs} "
                                        f"given {nonsinks}, Bs:{Bs}")
                            v_structure_found = True
            if v_structure_found == False:
                res_for_add.append((As, rk, nonsinks))
                found = True

    def flush_pending():
        if not pending:
            return
        ranktest.prime(_rank_cols(parameters['xvars'], G, As_, Bs_, list(nonsinks))
                       for As_, Bs_ in pending)
        for As_, Bs_ in pending:
            process_survivor(As_, Bs_)
        pending.clear()
        if hasattr(ranktest, 'clear'):
            ranktest.clear()

    for As in reversed(allSubsets):

        effective_ChildrenOfNonAtomicsSet = current_ChildrenOfNonAtomicsSet.copy()
        temp_set = As | {G.X_dict[t] for t in nonsinks}

        for cover in temp_set:
            effective_ChildrenOfNonAtomicsSet = effective_ChildrenOfNonAtomicsSet - G.findDescendants(cover, rigorous=False)

        Bs = setDifference(current_activeSet | effective_ChildrenOfNonAtomicsSet, As)
        observed_vars_in_As = {x.__str__() for x in As if x.is_observed}
        observed_vars_in_As_and_nonsinks = observed_vars_in_As.union(set(nonsinks))
        observed_vars_in_As_and_nonsinks = list(observed_vars_in_As_and_nonsinks)
        observed_vars_in_As_and_nonsinks_idx_in_local_adj = [G.x_list_for_local_Adj.index(x) for x in observed_vars_in_As_and_nonsinks]

        temp_local_adj = G.local_Adj[observed_vars_in_As_and_nonsinks_idx_in_local_adj, :][:, observed_vars_in_As_and_nonsinks_idx_in_local_adj]

        def check_dsu(adj):
            num_var = len(adj)
            dsu = DSU(num_var)
            for i in range(num_var):
                for j in range(num_var):
                    if i != j and (adj[i, j] != 0 or adj[j, i] != 0):
                        dsu.union(i, j)
            fa_set = set()
            for i in range(num_var):
                fa_set.add(dsu.find(i))
            if len(fa_set) == 1:
                return True
            else:
                return False

        if not check_dsu(temp_local_adj):
            continue

        if setLength(Bs) <= k - len(nonsinks):
            continue

        if G.containsCluster(As, nonsinks):
            continue

        if G.overlapPaCh(As):
            continue

        if G.MeassuredHasNonSinks(As, nonsinks):
            continue

        if G.checkNonSinksAreAsChildren(As, nonsinks):
            continue

        if G.parentCardinality(Bs) <= k - num_nonsinks:
            continue

        if batching:
            pending.append((As, Bs))
            if len(pending) >= BATCH_CHUNK:
                flush_pending()
        else:
            process_survivor(As, Bs)

    flush_pending()
    return (found, terminate, res_for_add)


def _at_k_serial(G, k, parameters, n_jobs=-1):
    """Upstream findClusters_at_k_mp with the Parallel() call replaced by a serial loop.
    Aggregation (collect all groups' res_for_add, then add) is kept identical."""
    from itertools import combinations
    LOGGER.info(f"Starting searchClusters k={k}...")
    global_terminate = True
    global_found = False
    found_deficiency = False

    input_list = []
    for num_nonsinks in range(k, -1, -1):
        temp_activeNonSink_ls = sorted(list(G.activeNonSinkSet), reverse=True)
        nonsinks_ls = list(combinations(temp_activeNonSink_ls, num_nonsinks))
        for nonsinks in nonsinks_ls:
            input_list.append(list(nonsinks).copy())

    output_list = [base.findClusters_at_k_by_nonsinks(G, k, nonsinks, parameters)
                   for nonsinks in input_list]

    for output in output_list:
        current_found_deficiency, current_terminate, res_for_add = output
        found_deficiency = found_deficiency or current_found_deficiency
        global_terminate = global_terminate and current_terminate
        for i in range(len(res_for_add)):
            G.addRankDefSet(res_for_add[i][0], res_for_add[i][1], used_nonsinks=res_for_add[i][2])

    if found_deficiency:
        G.determineClusters()
        found = G.confirmClusters()
        global_found = global_found or found
        if global_found:
            G.updateActiveSet()
            G.updateactiveNonSinkSet()
            M.display(G)
            return G, (global_found, global_terminate)

    return G, (global_found, global_terminate)


def _recboss_cpdag(X, discount, seed=1):
    """Stage-1 adjacency from the C BOSS in causal-learn CPDAG encoding (i -> j as
    A[j, i] = 1, A[i, j] = -1). Replaces the GES stage 1, which took 30+ minutes at p=42."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "upstream", "causal-get", "site"))
    import causalget as cg
    R = np.corrcoef(np.asarray(X, float), rowvar=False)
    dag = cg.boss(R, n=len(X), discount=discount, seed=seed)   # dag[i, j] = 1 means j -> i
    A = np.zeros(dag.shape)
    for i, j in zip(*np.nonzero(dag)):
        A[i, j] = 1
        A[j, i] = -1
    return A


MAX_GROUP = int(os.environ.get("RLCD_MAX_GROUP", 30))


def _recboss_auto(X, thres=3):
    """Escalate the stage-1 discount until the largest partition group is MAX_GROUP or
    smaller. RLCD's unfold phase is exponential in the latents per group, so a dense stage-1
    graph (strongly correlated scales at discount 2) stalls it. Returns the adjacency."""
    p = np.asarray(X).shape[1]
    fake_names = [f"X{i}" for i in range(p)]
    A = None
    for d in (2, 4, 8, 16, 32, 64):
        A = _recboss_cpdag(X, d)
        groups = base.getPartition(fake_names, np.abs(A), thres)
        biggest = max((len(g) for g in groups), default=0)
        print(f"  [stage1 auto] discount={d}: {len(groups)} groups, largest {biggest}",
              flush=True)
        if biggest <= MAX_GROUP:
            return A
    return A


def RLCD_gpu(data, ranktest_method=None, stage1_method="ges", stage1_discount=2.0, **kwargs):
    """RLCD with the batched serial sweep patched in for the duration of the call.

    stage1_method="recboss" computes the stage-1 adjacency with the C BOSS (seconds) and
    hands it to the unchanged partition logic by stubbing the GES call for this one run."""
    import causallearn.search.ScoreBased.GES as GESmod
    saved_mp = base.findClusters_at_k_mp
    saved_by = base.findClusters_at_k_by_nonsinks
    saved_ges = GESmod.ges
    base.findClusters_at_k_mp = _at_k_serial
    base.findClusters_at_k_by_nonsinks = _by_nonsinks_batched
    if stage1_method == "recboss":
        if str(stage1_discount) == "auto":
            A = _recboss_auto(data)
        else:
            A = _recboss_cpdag(data, float(stage1_discount))
        GESmod.ges = lambda *a, **k: {"G": type("G", (), {"graph": A})()}
        stage1_method = "ges"
    try:
        return base.RLCD(data, ranktest_method=ranktest_method,
                         stage1_method=stage1_method, **kwargs)
    finally:
        base.findClusters_at_k_mp = saved_mp
        base.findClusters_at_k_by_nonsinks = saved_by
        GESmod.ges = saved_ges


def RLCD_serial(data, ranktest_method=None, **kwargs):
    """Upstream logic exactly, but the sweep runs serially in this process. Used as the
    deterministic certification reference: the loky workers of the upstream mp path have
    per-process hash seeds, so their list(set(...)) column orders (and with them borderline
    test decisions) vary run to run."""
    saved_mp = base.findClusters_at_k_mp
    base.findClusters_at_k_mp = _at_k_serial
    try:
        return base.RLCD(data, ranktest_method=ranktest_method, **kwargs)
    finally:
        base.findClusters_at_k_mp = saved_mp
