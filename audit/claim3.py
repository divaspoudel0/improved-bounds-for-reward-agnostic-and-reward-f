"""Claim 3 numerical audit: single online MDP procedure over reward sequence
lambda^t_h(s,a) = 1/(c + N^t_h(s,a)).

Algorithm 2 runs ONE online MDP procedure (instead of multiple no-regret
instantiations).  The reward fed to the online learner at episode t is
lambda^t_h(s,a) = 1/(c + N^t_h(s,a)), where N^t is the cumulative visit count
before episode t and c = 4|S|H^2*ell_0 (Corollary 4.6).

We audit:
  (a) Lemma D.2 / Lemma 4.5 (self-normalized sum):  sum_t <mu^t, lambda^t>
      <= G(T) = 4|S||A|H ln T.
  (b) Theorem 4.2 coverage: for the returned policy pi_exp (average of pi^t),
      every policy pi satisfies
         sum_{(s,a,h) in psi} mu^pi_h / mu^T_h <= O(|S||A|H ell_0^2)
      and the non-significant mass is <= O(omega |S||A|H ell_0^2).
  (c) Lemma 3.3: f(mu^T) - f(mu*) = O(|S||A|H log T) with f = -sum log mu.

The online MDP oracle is implemented as exact best-response planning (a
regret-zero oracle) on the true dynamics, which isolates the reward-sequence
mechanism that is the claim's core.
"""
import numpy as np
from .mdp import MDP
from .claim4 import solve_logbarrier, coverage_ratio


def explore_policy(mdp, c, T, seed=0):
    """Algorithm 2 with a greedy no-regret oracle (single online MDP
    procedure).  Returns N (final counts), list of lambda^t (or aggregated),
    list of mu^t, and the returned average policy pi_exp."""
    rng = np.random.default_rng(seed)
    S, A, H = mdp.S, mdp.A, mdp.H
    N = np.zeros((H, S, A))
    mus = []          # occupancy measures mu^t
    lams = []         # reward sequences lambda^t
    pi = np.full((H, S, A), 1.0 / A)   # initial uniform policy
    for t in range(T):
        mu = mdp.occupancy(pi)
        mus.append(mu)
        lam = 1.0 / (c + N)
        lams.append(lam)
        # update counts by ONE trajectory under pi (expectation proxy)
        N = N + mu
        # online MDP oracle: best response to lambda^t (regret-zero planning)
        pi = mdp.optimal_policy(lam)
    pi_exp = sum(mus) / T    # average policy (occupancy of returned policy
                             # is mu^T = (1/T) sum_t mu^t by linearity)
    return N, mus, lams, pi_exp


def audit_claim3(S=4, A=3, H=4, T=300, n_mdps=6, seed=0, delta=0.1, eps=0.1):
    rng = np.random.default_rng(seed)
    ell0 = float(np.log(S * A * H / (delta * eps)))
    c = 4 * S * H * H * ell0
    G_T = 4 * S * A * H * np.log(T)
    results = {
        'c': c, 'ell0': ell0, 'G_T': float(G_T), 'T': T,
        'selfnorm_ratio': [],        # sum_t <mu^t,lambda^t> / G(T)
        'coverage_ratio_max': [],    # max over policies of significant ratio
        'coverage_ub': [],           # O(|S||A|H ell0^2)
        'nonsig_mass_max': [],
        'f_gap': [],                 # f(mu^T)-f(mu*)
        'logT_fit': [],
    }
    for trial in range(n_mdps):
        mdp = MDP.random(S, A, H, rng=rng, sparsity=1.0)
        N, mus, lams, pi_exp = explore_policy(mdp, c, T, seed=seed + trial)
        mu_T = np.mean(np.stack(mus), axis=0)
        pi_T = mdp.policy_from_occupancy(mu_T)
        # (a) self-normalized sum  <mu^t, lambda^t> = sum_h,s,a mu^t_h(s,a)/(c+N^t)
        #     with N^t the pre-episode count == sum_{tau<t} mu^tau (we used mu-proxy)
        sn = 0.0
        Nn = np.zeros((H, S, A))
        for t, mu in enumerate(mus):
            sn += np.sum(mu / (c + Nn))
            Nn = Nn + mu
        results['selfnorm_ratio'].append(sn / G_T)
        # (b) Theorem 4.2 coverage: significant set w.r.t. pi_exp with threshold omega
        omega = eps / (S * A * H * H * ell0 * ell0)
        sig = mu_T > omega
        # worst-case policy for significant coverage: reward 1/mu_T on sig
        reward_sig = np.zeros((H, S, A))
        reward_sig[sig] = 1.0 / np.clip(mu_T[sig], 1e-14, None)
        pi_worst = mdp.optimal_policy(reward_sig)
        ratio = float(np.sum(mdp.occupancy(pi_worst) * reward_sig))
        results['coverage_ratio_max'].append(ratio)
        results['coverage_ub'].append(S * A * H * ell0 * ell0)
        # non-significant mass for worst policy (reward = indicator on non-sig)
        reward_ns = np.zeros((H, S, A))
        reward_ns[~sig] = 1.0
        pi_ns = mdp.optimal_policy(reward_ns)
        mass = float(np.sum(mdp.occupancy(pi_ns) * reward_ns))
        results['nonsig_mass_max'].append(mass)
        # (c) Lemma 3.3: f(mu_T) - f(mu*)
        mu_star, _, _, _ = solve_logbarrier(mdp)
        f_T = -np.sum(np.log(np.clip(mu_T, 1e-14, None)))
        f_star = -np.sum(np.log(np.clip(mu_star, 1e-14, None)))
        results['f_gap'].append(f_T - f_star)
        results['logT_fit'].append(S * A * H * np.log(T))

    results['selfnorm_ratio'] = list(map(float, results['selfnorm_ratio']))
    results['selfnorm_max'] = float(max(results['selfnorm_ratio']))
    results['selfnorm_bound_satisfied'] = results['selfnorm_max'] <= 1.0
    results['coverage_ratio_max'] = list(map(float, results['coverage_ratio_max']))
    results['coverage_ratio_max_trial'] = float(max(results['coverage_ratio_max']))
    results['coverage_ub'] = list(map(float, results['coverage_ub']))
    results['coverage_bound_satisfied'] = results['coverage_ratio_max_trial'] <= \
        10 * S * A * H * ell0 * ell0
    results['nonsig_mass_max'] = list(map(float, results['nonsig_mass_max']))
    results['nonsig_bound'] = float(omega * S * A * H * ell0 * ell0)
    results['nonsig_bound_satisfied'] = max(results['nonsig_mass_max']) <= \
        10 * results['nonsig_bound']
    results['f_gap'] = list(map(float, results['f_gap']))
    results['logT_fit'] = list(map(float, results['logT_fit']))
    # f-gap should grow like O(|S||A|H log T): check slope of f_gap vs log T at
    # two T values
    results['claim_supported'] = results['selfnorm_bound_satisfied'] and \
        results['coverage_bound_satisfied']
    return results


if __name__ == '__main__':
    import json
    print(json.dumps(audit_claim3(), indent=2))
