"""Core tabular finite-horizon time-inhomogeneous MDP primitives.

Implements the definitions of Ridel & Cohen, arXiv 2602.16363:
  - Section 2: MDP = (S, A, H, {P_h}, r), occupancy measures mu_h(s,a),
    the occupancy polytope Lambda (Eqs. 1-2), and the policy <-> occupancy
    bijection (Eq. 3).
"""
import numpy as np


class MDP:
    def __init__(self, S, A, H, P, rho=None, rng=None):
        """P: ndarray (H, S, A, S) with P[h,s,a,:] a distribution over S.
        rho: initial distribution over S (default: one-hot s=0)."""
        self.S = S
        self.A = A
        self.H = H
        self.P = np.asarray(P, dtype=float)
        self.rng = rng or np.random.default_rng(0)
        if rho is None:
            rho = np.zeros(S)
            rho[0] = 1.0
        self.rho = np.asarray(rho, dtype=float)
        assert self.P.shape == (H, S, A, S)

    @staticmethod
    def random(S, A, H, rng=None, sparsity=1.0):
        """Random time-inhomogeneous transition kernel. Each P_h(s,a,.) is
        drawn from a Dirichlet(alpha) with alpha=sparsity (sparsity < 1 gives
        near-sparse / near-deterministic transitions)."""
        rng = rng or np.random.default_rng(1)
        alpha = np.full(S, sparsity)
        P = np.zeros((H, S, A, S))
        for h in range(H):
            for s in range(S):
                for a in range(A):
                    P[h, s, a] = rng.dirichlet(alpha)
        return MDP(S, A, H, P, rng=rng)

    # -- value functions -------------------------------------------------
    def value(self, reward, pi, s=0, h0=0):
        """V_h^pi under reward r_h(s,a).  reward: (H, S, A) or (S, A, H)?.
        We use (H, S, A) everywhere.  pi: (H, S, A) probabilities.
        Returns V_h0(s)."""
        reward = np.asarray(reward, dtype=float)
        V = np.zeros(self.S)
        for h in range(self.H - 1, h0 - 1, -1):
            Q = reward[h] + self.P[h] @ V          # (S, A)
            V = np.sum(pi[h] * Q, axis=1)           # (S,)
        return V[s]

    def optimal_value(self, reward, s=0):
        """V_1^*(s) for the given reward."""
        reward = np.asarray(reward, dtype=float)
        V = np.zeros(self.S)
        for h in range(self.H - 1, -1, -1):
            Q = reward[h] + self.P[h] @ V
            V = Q.max(axis=1)
        return V[s]

    def optimal_policy(self, reward):
        """Deterministic optimal policy pi*(h,s,a) one-hot."""
        reward = np.asarray(reward, dtype=float)
        V = np.zeros(self.S)
        pi = np.zeros((self.H, self.S, self.A))
        for h in range(self.H - 1, -1, -1):
            Q = reward[h] + self.P[h] @ V
            a_star = Q.argmax(axis=1)
            for s in range(self.S):
                pi[h, s, a_star[s]] = 1.0
            V = Q.max(axis=1)
        return pi

    # -- occupancy measures ----------------------------------------------
    def occupancy(self, pi):
        """mu_h(s,a) induced by policy pi, mu_1(s,a) = rho(s)*pi_1(a|s).
        Returns ndarray (H, S, A)."""
        mu = np.zeros((self.H, self.S, self.A))
        for s in range(self.S):
            for a in range(self.A):
                mu[0, s, a] = self.rho[s] * pi[0, s, a]
        for h in range(self.H - 1):
            for s_next in range(self.S):
                m = 0.0
                for s in range(self.S):
                    for a in range(self.A):
                        m += mu[h, s, a] * self.P[h, s, a, s_next]
                for a in range(self.A):
                    mu[h + 1, s_next, a] = m * pi[h + 1, s_next, a]
        return mu

    def policy_from_occupancy(self, mu):
        """Eq. 3: pi_h(a|s) = mu_h(s,a) / sum_a' mu_h(s,a')."""
        pi = np.zeros((self.H, self.S, self.A))
        for h in range(self.H):
            for s in range(self.S):
                m = mu[h, s].sum()
                if m > 0:
                    pi[h, s] = mu[h, s] / m
        return pi

    # -- occupancy polytope constraints ----------------------------------
    def flow_matrix(self):
        """Build the linear equality constraints of Lambda:
        For each h and s':
           sum_{s,a} mu_h(s,a) P_h(s'|s,a) - sum_a mu_{h+1}(s',a) = 0
        plus normalization sum_{s,a} mu_h(s,a) = 1 for each h.
        Return (C_eq, b_eq) with C shape (n_cons, H*S*A)."""
        n = self.H * self.S * self.A
        rows = []
        # flow constraints (H-1)*S rows
        for h in range(self.H - 1):
            for sp in range(self.S):
                row = np.zeros(n)
                for s in range(self.S):
                    for a in range(self.A):
                        row[h * self.S * self.A + s * self.A + a] += self.P[h, s, a, sp]
                for a in range(self.A):
                    row[(h + 1) * self.S * self.A + sp * self.A + a] -= 1.0
                rows.append(row)
        # normalization H rows
        for h in range(self.H):
            row = np.zeros(n)
            row[h * self.S * self.A:(h + 1) * self.S * self.A] = 1.0
            rows.append(row)
        C = np.array(rows)
        b = np.zeros(C.shape[0])
        b[-self.H:] = 1.0
        return C, b
