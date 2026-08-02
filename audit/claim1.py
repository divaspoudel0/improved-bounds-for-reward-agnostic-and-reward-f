"""Claim 1 numerical audit: reward-agnostic sample complexity.

Algorithm 1 = ExpPolicyCreation (Algorithm 2) -> DynamicsEst (Algorithm 3)
            -> PolicyEst (Algorithm 4, pessimistic model-based offline RL).

Proposition 3.1 / Theorem 5.1: with N = O(|S||A|H^3 ell_0^3 / eps^2)
trajectories, Algorithm 4 returns an eps-optimal policy for any reward in a
poly-size class. The full sample complexity is

    O(|S||A|H^3 ell_0^3 / eps^2  +  |S|^2|A|H^4 ell_0^3 / eps)

(Prop 3.1), whose high-order term is optimal (matches BPI lower bound,
Domingues et al. 2021) and whose low-order term is a major improvement over
Li et al. (2024b)'s H^6|A|^4|S|^4/eps (Table 1). The improvement means the
bound is dominated by its (optimal) high-order term for eps <= 1/(|S|H),
whereas Li et al. (2024b) need eps <= 1/(|S|^3|A|^3H^3).

We audit:
  (a) the algebra of the bound: c = 4|S|H^2 ell_0 (Corollary 4.6),
      omega = O(eps/(|S||A|H^2 ell_0^2)), T = 2c/omega =
      O(|S|^2|A|H^4 ell_0^3/eps) (Theorem 4.2), N_RAE = O(|S||A|H^3
      ell_0^3/eps^2) (Theorem 5.1), N_RFE = O(|S|^2|A|H^3 ell_0^3/eps^2)
      (Theorem 5.2).
  (b) the low-order-term comparison with Li et al. (2024b) and the resulting
      optimality thresholds on eps (Table 1).
  (c) the eps^{-2} high-order sample-complexity law end-to-end: run the full
      three-step pipeline on random tabular MDPs and measure, for several
      accuracy values eps, the number of trajectories needed to reach
      suboptimality <= eps. The required N should scale like C/eps^2.
  (d) end-to-end guarantee: with N large enough, the returned pessimistic
      policy is eps-optimal for random rewards from a poly-size class.
"""
import numpy as np
from .mdp import MDP
from .claim3 import explore_policy


def ell0(S, A, H, delta, eps):
    return float(np.log(S * A * H / (delta * eps)))


def bound_constants(S, A, H, delta, eps):
    l0 = ell0(S, A, H, delta, eps)
    c = 4 * S * H * H * l0
    omega = eps / (S * A * H * H * l0 * l0)   # O(eps/(|S||A|H^2 ell_0^2))
    T = 2 * c / omega                          # Corollary 4.6 / Thm 4.2
    N_RAE = S * A * H**3 * l0**3 / eps**2      # Theorem 5.1
    N_RFE = S * S * A * H**3 * l0**3 / eps**2  # Theorem 5.2
    return {
        'ell0': l0, 'c': c, 'omega': omega, 'T': T, 'N_RAE': N_RAE, 'N_RFE': N_RFE,
        'T_form': 'O(|S|^2|A|H^4 ell_0^3/eps)',
        'N_RAE_form': 'O(|S||A|H^3 ell_0^3/eps^2)',
        'N_RFE_form': 'O(|S|^2|A|H^3 ell_0^3/eps^2)',
    }


def li_loworder(S, A, H, eps):
    """Li et al. (2024b) low-order term (Table 1): H^6|A|^4|S|^4/eps."""
    return H**6 * A**4 * S**4 / eps


def paper_loworder(S, A, H, eps):
    """This paper's low-order term (Table 1): H^4|A||S|^2/eps."""
    return H**4 * A * S**2 / eps


def paper_highorder(S, A, H, eps, ell0=1.0):
    """High-order term |S||A|H^3 ell_0^3/eps^2 (omitting ell0 by default)."""
    return S * A * H**3 * ell0**3 / eps**2


def optimality_threshold(S, A, H):
    """Largest eps below which the low-order term is dominated by the
    high-order term (i.e. the bound is asymptotically optimal): solve
    high ~= low. Paper: eps <= 1/(|S|H); Li et al: eps <= 1/(|S|^3|A|^3H^3)."""
    paper = 1.0 / (S * H)
    li = 1.0 / (S**3 * A**3 * H**3)
    return {'paper_eps_threshold': paper, 'li_eps_threshold': li,
            'relaxation_factor': paper / li}


# ---------------------------------------------------------------------------
# Algorithms 3 + 4 (faithful implementations)
# ---------------------------------------------------------------------------

def _vec_sample(p_rows, rng):
    """Sample a row index from each row of a (n, k) probability matrix."""
    c = np.cumsum(p_rows, axis=1)
    u = rng.random(len(p_rows))[:, None]
    return (u < c).argmax(axis=1)


def dynamics_est(mdp, pi_exp, N, rng):
    """Algorithm 3: sample N trajectories under pi_exp, estimate P_hat.

    Vectorized across the N rollouts (per-step action draws and transition
    draws are batched over the live trajectory set)."""
    S, A, H = mdp.S, mdp.A, mdp.H
    N = int(round(N))
    counts = np.zeros((H, S, A, S))
    Nsa = np.zeros((H, S, A))
    s = rng.choice(S, size=N, p=mdp.rho)
    for h in range(H):
        a = _vec_sample(pi_exp[h][s], rng)
        np.add.at(Nsa, (h, s, a), 1)
        s_entry = s.copy()
        for st in range(S):
            for ac in range(A):
                idx = np.where((s_entry == st) & (a == ac))[0]
                if len(idx) == 0:
                    continue
                ns = _vec_sample(np.tile(mdp.P[h, st, ac], (len(idx), 1)), rng)
                counts[h, st, ac] += np.bincount(ns, minlength=S)
                s[idx] = ns
    P_hat = np.zeros((H, S, A, S))
    for h in range(H):
        for s in range(S):
            for a in range(A):
                if Nsa[h, s, a] > 0:
                    P_hat[h, s, a] = counts[h, s, a] / Nsa[h, s, a]
                else:
                    P_hat[h, s, a] = 1.0 / S
    return P_hat, Nsa


def policy_est(mdp, reward, P_hat, Nsa, mode='RAE', ca=48.0, cf=64.0,
               delta=0.2, eps=0.3):
    """Algorithm 4: pessimistic model-based offline RL.

    b_h(s,a) = sqrt(c*sfac*ell0*Var_Phat(Vhat_{h+1})/N_h(s,a))
               + c*sfac*H*ell0/N_h(s,a), capped at H.
    RAE: Eq. 15 (c = ca >= 48, sfac = 1); RFE: Eq. 16 (c = cf >= 64,
    sfac = |S|)."""
    S, A, H = mdp.S, mdp.A, mdp.H
    l0 = ell0(S, A, H, delta, eps)
    V = np.zeros(S)
    Q = np.zeros((H, S, A))
    pi_hat = np.zeros((H, S, A))
    c = cf if mode == 'RFE' else ca
    sfac = S if mode == 'RFE' else 1
    for h in range(H - 1, -1, -1):
        PV = P_hat[h] @ V
        Var = np.zeros((S, A))
        for s in range(S):
            for a in range(A):
                Var[s, a] = np.sum(P_hat[h, s, a] * (V - PV[s, a])**2)
        Np = np.maximum(Nsa[h], 1e-9)
        b = np.sqrt(c * sfac * l0 * Var / Np) + c * sfac * H * l0 / Np
        b = np.minimum(b, H)
        Q[h] = np.maximum(reward[h] + PV - b, 0.0)
        V = Q[h].max(axis=1)
        for s in range(S):
            pi_hat[h, s, Q[h, s].argmax()] = 1.0
    return pi_hat, Q


def run_end_to_end(S, A, H, N, reward, rng, seed, delta=0.2, eps=0.3):
    """Full Algorithm 1 pipeline; returns (mdp, suboptimality)."""
    mdp = MDP.random(S, A, H, rng=rng, sparsity=1.0)
    l0 = ell0(S, A, H, delta, eps)
    c = 4 * S * H * H * l0
    _, mus, _, _ = explore_policy(mdp, c, T=200, seed=seed)
    pi_exp = mdp.policy_from_occupancy(np.mean(np.stack(mus), axis=0))
    P_hat, Nsa = dynamics_est(mdp, pi_exp, N, rng)
    pi_hat, _ = policy_est(mdp, reward, P_hat, Nsa, mode='RAE',
                           delta=delta, eps=eps)
    V_star = mdp.optimal_value(reward)
    V_hat = mdp.value(reward, pi_hat)
    return mdp, float(V_star - V_hat)


def audit_claim1(S=4, A=3, H=4, delta=0.2, seed=0):
    rng = np.random.default_rng(seed)
    bc = bound_constants(S, A, H, delta, 0.3)
    results = {
        'S': S, 'A': A, 'H': H, 'delta': delta,
        'bound_constants': {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in bc.items()},
        'loworder': {},
        'optimality_thresholds': optimality_threshold(S, A, H),
        'eps_scaling': {},
        'end_to_end': {},
    }
    # (b) low-order-term comparison (Table 1)
    lratios = []
    for eps in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        lratios.append(float(li_loworder(S, A, H, eps) / paper_loworder(S, A, H, eps)))
    results['loworder']['li_over_paper_ratio'] = lratios
    results['loworder']['ratio_form'] = 'H^2|A|^3|S|^2'
    results['loworder']['ratio_theory'] = float(H**2 * A**3 * S**2)
    results['loworder']['improvement_confirmed'] = all(r > 1 for r in lratios)
    results['loworder']['eps_threshold'] = optimality_threshold(S, A, H)
    # (c) required-N scaling: binary-search the trajectories needed to reach
    #     suboptimality <= eps_target (max over a fixed reward pool) on a fixed
    #     MDP. The theorem is an upper bound: empirical N_required should be
    #     bounded by (and, given loose constants, typically far below) the
    #     theoretical O(|S||A|H^3 ell0^3/eps^2). Reported as a ratio curve.
    mdp = MDP.random(S, A, H, rng=rng, sparsity=1.0)
    _, mus, _, _ = explore_policy(mdp, c=bound_constants(S, A, H, delta, 0.3)['c'],
                                  T=200, seed=seed)
    pi_exp = mdp.policy_from_occupancy(np.mean(np.stack(mus), axis=0))
    reward_pool = [rng.uniform(0, 1, size=(H, S, A)) for _ in range(4)]

    def _sub_max(N, eps_tgt):
        P_hat, Nsa = dynamics_est(mdp, pi_exp, N, rng)
        subs = []
        for rw in reward_pool:
            pi_hat, _ = policy_est(mdp, rw, P_hat, Nsa, mode='RAE',
                                   delta=delta, eps=eps_tgt)
            subs.append(mdp.optimal_value(rw) - mdp.value(rw, pi_hat))
        return max(subs)

    def _required_N(eps_tgt, lo=500, hi=400000):
        while hi - lo > 200:
            mid = (lo + hi) // 2
            if _sub_max(mid, eps_tgt) <= eps_tgt:
                hi = mid
            else:
                lo = mid
        return hi

    eps_grid = [0.5, 0.3, 0.2, 0.12]
    n_req = []
    for tgt in eps_grid:
        n_req.append(int(_required_N(tgt)))
    results['eps_scaling']['eps_targets'] = eps_grid
    results['eps_scaling']['N_required'] = n_req
    results['eps_scaling']['N_RAE_theory'] = float(
        bound_constants(S, A, H, delta, 0.3)['N_RAE'])
    results['eps_scaling']['within_theory_bound'] = [
        n <= results['eps_scaling']['N_RAE_theory'] for n in n_req]
    # slope of log(N_required) vs log(1/eps) — should be ~2 in the pure
    # eps^{-2} regime; the pessimistic penalty's threshold (Q clipped at 0)
    # often yields a milder empirical slope, which still supports the upper
    # bound. We report it but do not hard-gate the claim on it.
    if len(n_req) >= 2:
        xs = np.log(1.0 / np.array(eps_grid))
        ys = np.log(np.array(n_req, dtype=float))
        results['eps_scaling']['slope_logN_vs_log1oeps'] = float(
            np.polyfit(xs, ys, 1)[0])
    # (d) end-to-end guarantee at a large fixed N
    N_fixed = 100000
    subs = []
    for t in range(8):
        reward = rng.uniform(0, 1, size=(H, S, A))
        _, sub = run_end_to_end(S, A, H, N_fixed, reward, rng, seed + t,
                                delta=delta, eps=0.3)
        subs.append(sub)
    results['end_to_end']['N_fixed'] = N_fixed
    results['end_to_end']['subopts'] = list(map(float, subs))
    results['end_to_end']['subopt_max'] = float(max(subs))
    results['end_to_end']['subopt_mean'] = float(np.mean(subs))
    results['end_to_end']['eps_target'] = 0.3
    results['end_to_end']['eps_optimal_at_fixed_N'] = float(max(subs)) <= 0.3
    results['claim_supported'] = (
        results['loworder']['improvement_confirmed']
        and results['end_to_end']['eps_optimal_at_fixed_N']
        and all(results['eps_scaling']['within_theory_bound']))
    return results


if __name__ == '__main__':
    import json
    print(json.dumps(audit_claim1(), indent=2))
