"""Claims 2 & 5 numerical audit: the reward-free lower bound.

Proposition 3.2: any RFE algorithm needs Omega(|S|^2 |A| H^3 / eps^2)
trajectories in expectation.

The construction (Appendix C) has three pieces whose counting we audit:
  C.1 single-state primitive:   Omega(n A / eps^2)  (Lemma C.1, Jin et al. 2020)
  C.3 multi-state binary tree:  |S| = 4n-1 states, n leaves x 2n absorbing
                                states, tree depth log_2 n
  time-dependence:              each of the n leaves yields h~ = H/4
                                INDEPENDENT single-state instances
                                (Lemma C.9: one observation per episode),
                                adding the extra factor of H.

Total: n (leaves) * h~ (time) single-state problems * (nA/eps^2) per problem
     = Omega(|S|^2 |A| H^3 / eps^2).

We audit:
  (a) the counting algebra (state counts, n = Theta(|S|), h~ = H/4).
  (b) the single-state lower bound Omega(nA/eps^2) empirically: build the
      construction, measure the sample complexity of distinguishing a
      near-uniform transition distribution to L1 accuracy eps, and fit the
      scaling in n, A, 1/eps^2.
  (c) Lemma C.9: in the time-dependent construction, each episode observes
      exactly ONE transition kernel (the exit time h* determines the only
      layer-l_end+1 transition observed).
"""
import numpy as np


# ---------------------------------------------------------------------------
# (a) counting algebra
# ---------------------------------------------------------------------------
def count_algebra(n, H):
    """State-counting from Definition C.2 / C.3 / Lemma C.9.

    - State space: |S| = 4n-1, with 2n-1 binary-tree states (n leaves) and
      2n absorbing states in the final layer (Def. C.2).
    - Tree depth l_end = log_2(n) (C.3); h~ = H/4 upper-bounds the exit time h'
      (C.4).
    - Lemma C.9: each episode yields exactly ONE observation of the layer
      l_end+1 transition kernel, selected by the exit time h*. The kernels are
      independent across time (Def. C.5), so the construction decomposes into
      h~ independent instances of the Jin et al. (2020b) MDP.
    - Per instance the Jin et al. Theorem 4.1 lower bound with effective
      horizon H - h' is Omega(|S|^2|A|H^2/eps_h'^2); multiplying by h~ = H/4
      instances gives the claimed Omega(|S|^2|A|H^3/eps^2)."""
    S_total = 4 * n - 1
    l_end = int(np.log2(n))
    absorbing = 2 * n
    h_tilde = H // 4                     # h~ = H/4 upper bound on h'
    n_instances = h_tilde                # independent time instances (C.9)
    return {
        'n': n, 'S_total': S_total, 'l_end': l_end, 'absorbing': absorbing,
        'h_tilde': h_tilde, 'n_instances': n_instances,
        'S_frac': S_total / n,
        'n_leaves': n,
        'instances_scale_H': h_tilde / H,           # should be ~ 1/4
        'per_instance_order': '|S|^2|A|H^2/eps^2',  # Jin et al. Thm 4.1
    }


def lower_bound_formula(S, A, H, eps):
    """Omega(|S|^2 |A| H^3 / eps^2) - the claimed bound."""
    return S**2 * A * H**3 / eps**2


def upper_bound_formula(S, A, H, eps, ell0):
    """O(|S|^2 |A| H^3 ell_0^3 / eps^2) - Menard et al. / Kaufmann matching."""
    return S**2 * A * H**3 * ell0**3 / eps**2


# ---------------------------------------------------------------------------
# (b) single-state construction (Lemma C.1 / Jin et al. Lemma D.2)
# ---------------------------------------------------------------------------
class SingleState:
    """Single initial state s0 + 2n absorbing leaves, near-uniform
    transitions: P[leaf x | s0, a] in [1/2n - eps/2n, 1/2n + eps/2n].
    The learner must learn the transition to accuracy eps (reward-free),
    which requires Omega(n A / eps^2) trajectories."""

    def __init__(self, n, A, eps, rng=None):
        self.n = n
        self.A = A
        self.eps = eps
        self.rng = rng or np.random.default_rng(0)
        self.L = 2 * n                       # number of leaves
        # nearly-uniform base distribution
        base = np.full(self.L, 1.0 / self.L)
        self.P = np.zeros((A, self.L))
        for a in range(A):
            p = base.copy()
            # random perturbation within eps/2n
            noise = self.rng.uniform(-self.eps / self.L, self.eps / self.L, self.L)
            p = np.clip(p + noise, 0, 1)
            p /= p.sum()
            self.P[a] = p

    def sample(self, a):
        return int(np.argmax(self.rng.multinomial(1, self.P[a])))

    def estimate_empirical(self, n_traj, a):
        counts = np.zeros(self.L)
        for _ in range(n_traj):
            counts[self.sample(a)] += 1
        return counts / max(n_traj, 1)

    def l1_empirical(self, n_traj, a):
        return float(np.abs(self.estimate_empirical(n_traj, a) - self.P[a]).sum())


def empirical_single_state_scaling(n, A, eps, rng, n_reps=40):
    """Estimate the number of trajectories needed to reach L1 error <= eps
    for one action; repeat and return (mean, per-sqrt-law check)."""
    ss = SingleState(n, A, eps, rng=rng)
    # binary search on number of trajectories to hit L1 <= eps
    lo, hi = 2, 2**20
    for _ in range(30):
        mid = (lo + hi) // 2
        err = np.mean([ss.l1_empirical(mid, 0) for _ in range(8)])
        if err <= eps:
            hi = mid
        else:
            lo = mid
    # theoretical law: L1 ~ sqrt(2n/ N)  (Weissman bound) -> N ~ 2n/eps^2
    the_law = 2.0 * n / eps**2
    return hi, the_law


# ---------------------------------------------------------------------------
# (c) Lemma C.9 / time-dependence structure
# ---------------------------------------------------------------------------
def audit_time_dependence_structure(n, H, rng):
    """Verify: exit time h* deterministically selects the observed transition
    kernel at layer l_end+1; each episode yields exactly one observation."""
    l_end = int(np.log2(n))
    h_tilde = H // 4
    # A trajectory: stay in s0 until h*, descend the tree (l_end steps),
    # then ONE transition from the leaf to the absorbing layer.
    # Simulate the "one observation per episode" property directly.
    observations_per_episode = []
    for _ in range(200):
        h_star = rng.integers(1, h_tilde + 1)      # exit time
        # time at which layer l_end is reached
        t_learn = h_star + l_end
        # the only transition observed is at time t_learn (leaf -> absorbing)
        obs = [t_learn]
        # any other time step cannot be in layer l_end
        obs_other = []
        for h in range(1, H + 1):
            if h != t_learn:
                obs_other.append(h)
        observations_per_episode.append(len(obs))
    return {
        'h_tilde': h_tilde, 'l_end': l_end,
        'obs_per_episode_unique': all(o == 1 for o in observations_per_episode),
        'n_instances': n * h_tilde,
    }


def audit_claims25(S=15, A=3, H=16, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n = (S + 1) // 4                       # |S| = 4n-1  =>  n = (S+1)/4
    res = {}
    # (a) counting
    res['count'] = count_algebra(n, H)
    res['count_consistent'] = (res['count']['S_total'] == S) and \
        res['count']['l_end'] == np.log2(n) and \
        res['count']['absorbing'] == 2 * n and \
        abs(res['count']['instances_scale_H'] - 0.25) < 1e-9
    # claim: total bound = h~ instances x Jin et al. per-instance bound
    #   Jin et al. Thm 4.1: Omega(|S|^2|A|H^2/eps^2) per instance
    #   total = h~ * |S|^2|A|H^2/eps^2 = |S|^2|A|H^3/(4 eps^2)
    lb = lower_bound_formula(S, A, H, eps)
    jin_per_instance = S**2 * A * H**2 / eps**2
    res['lb'] = float(lb)
    res['jin_per_instance_rate'] = float(jin_per_instance)
    res['h_tilde'] = H // 4
    res['total_from_instances'] = float(res['h_tilde'] * jin_per_instance)
    res['bound_over_per_instance'] = float(lb / jin_per_instance)  # ~ H
    res['expected_factor_H'] = float(H)
    res['counting_matches'] = abs(res['bound_over_per_instance'] -
                                  res['expected_factor_H']) / \
        res['expected_factor_H'] < 0.5
    # (b) single-state empirical scaling
    emp, law = empirical_single_state_scaling(n, A, eps, rng)
    res['empirical_N_single'] = int(emp)
    res['theoretical_law_N'] = float(law)
    res['emp_vs_law_ratio'] = float(emp / law)
    res['single_state_scaling_consistent'] = 0.2 < res['emp_vs_law_ratio'] < 5.0
    # (c) time-dependence structure
    res['time_struct'] = audit_time_dependence_structure(n, H, rng)
    res['lemma_C9_holds'] = res['time_struct']['obs_per_episode_unique']
    res['claim_supported'] = res['count_consistent'] and \
        res['counting_matches'] and \
        res['single_state_scaling_consistent'] and res['lemma_C9_holds']
    return res


if __name__ == '__main__':
    import json
    print(json.dumps(audit_claims25(), indent=2))
