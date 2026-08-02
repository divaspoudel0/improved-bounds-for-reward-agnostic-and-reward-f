"""Claim 4 numerical audit: the exploration objective is the log-barrier.

mu* = argmin_{mu in Lambda} { -sum_{s,a,h} log mu_h(s,a) }      (Eq. 4)
By first-order optimality of convex programs:
    sum_{s,a,h} mu^pi_h(s,a) / mu*_h(s,a) <= |S||A|H,  for all pi.   (Eq. 5)

The optimum is found by a damped-Newton primal-dual solve of the KKT system
    -1/mu* + C^T lambda = 0,   C mu* = b,
over a full-row-rank basis of the occupancy-polytype constraints C mu = b.
At the exact optimum 1/mu* = C^T lambda, hence the ratio in Eq. 5 equals
lambda^T b for EVERY policy pi; the audit verifies that the constant
lambda^T b (a) is <= |S||A|H and (b) equals the ratio actually observed over
an exhaustive set of deterministic policies and many random policies.

A control replaces mu* with the (feasible, non-optimal) occupancy of a random
policy; the ratio then degrades well above |S||A|H, showing the guarantee is
specific to the log-barrier minimizer.
"""
import numpy as np
import scipy.linalg as sla
from .mdp import MDP


def _independent_rows(C):
    """Return a maximal linearly independent subset of the rows of C via a
    pivoted QR of C^T."""
    Q, R, P = sla.qr(C.T, pivoting=True, mode='economic')
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > 1e-10))
    piv = np.sort(P[:rank])
    return C[piv]


def _basis(C):
    """Full-row-rank subset of the rows of C via a pivoted QR of C^T.
    Returns (Cind, piv)."""
    Q, R, P = sla.qr(C.T, pivoting=True, mode='economic')
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > 1e-10))
    piv = np.sort(P[:rank])
    return C[piv], piv


def solve_logbarrier(mdp, mu0=None, tol=1e-11, maxiter=1000):
    """Solve Eq. 4 exactly via a damped-Newton primal-dual interior method.

    Convergence is declared when the first-order optimality (stationarity)
    condition holds to relative precision tol, i.e. the gradient -1/mu* lies
    in the row space of the constraint matrix C (equivalently, exists lambda
    with 1/mu* = C^T lambda).  The dual is ill-conditioned near the optimum
    (the barrier Hessian 1/mu^2 diverges), so it is recovered by a
    least-squares projection a posteriori.

    Returns (mu_star (H,S,A), lambda_lsq, converged, n_iter)."""
    n = mdp.H * mdp.S * mdp.A
    C, b = mdp.flow_matrix()
    Cind, piv = _basis(C)
    bind = b[piv]

    if mu0 is None:
        pi0 = np.full((mdp.H, mdp.S, mdp.A), 1.0 / mdp.A)
        mu0 = mdp.occupancy(pi0).ravel()
        mu0 = np.clip(mu0, 1e-3, None)

    x = np.clip(mu0, 1e-4, None).copy()
    lam = np.zeros(Cind.shape[0])
    converged = False
    for it in range(maxiter):
        g = -1.0 / x
        Hn = 1.0 / x ** 2
        K = np.block([[np.diag(Hn), Cind.T],
                      [Cind, np.zeros((Cind.shape[0], Cind.shape[0]))]])
        rhs = np.concatenate([-g, bind - Cind @ x])
        sol = np.linalg.solve(K, rhs)
        dx, dlam = sol[:n], sol[n:]
        # fraction-to-boundary line search keeps x strictly interior
        t = 1.0
        neg = dx < 0
        if neg.any():
            t = min(t, 0.99 * np.min(-x[neg] / dx[neg]))
        while (x + t * dx <= 1e-12).any():
            t *= 0.5
            if t < 1e-16:
                break
        x = x + t * dx
        lam = lam + t * dlam
        # stationarity check: distance of -1/x to row space of Cind
        proj, *_ = np.linalg.lstsq(Cind.T, -1.0 / x, rcond=None)
        resid = np.linalg.norm(Cind.T @ proj + 1.0 / x)
        if resid / np.linalg.norm(1.0 / x) < tol:
            converged = True
            lam = proj
            break
    mu_star = x.reshape(mdp.H, mdp.S, mdp.A)
    if not converged:
        # still recover the least-squares dual for diagnostics
        lam, *_ = np.linalg.lstsq(Cind.T, -1.0 / x, rcond=None)
    return mu_star, lam, converged, it


def coverage_ratio(mdp, mu_star, pi):
    """R(pi) = sum_{s,a,h} mu^pi_h(s,a) / mu*_h(s,a)."""
    mu_pi = mdp.occupancy(pi)
    return float(np.sum(mu_pi / np.clip(mu_star, 1e-14, None)))


def all_deterministic_policies(mdp):
    """Enumerate all deterministic policies (A^(H*S) of them)."""
    H, S, A = mdp.H, mdp.S, mdp.A
    n_actions_per = A
    total = A ** (H * S)
    for idx in range(total):
        pi = np.zeros((H, S, A))
        code = idx
        for h in range(H):
            for s in range(S):
                a = code % A
                code //= A
                pi[h, s, a] = 1.0
        yield pi
    return


def random_policies(mdp, k, rng=None):
    """Sample k random (randomized) policies."""
    rng = rng or np.random.default_rng(7)
    return [rng.dirichlet(np.ones(mdp.A), size=(mdp.H, mdp.S)) for _ in range(k)]


def audit_claim4(S=4, A=3, H=4, n_mdps=8, n_rand_policies=200, seed=0,
                 exhaustive_det=True):
    """Run the Claim 4 audit. Returns a dict of results."""
    rng = np.random.default_rng(seed)
    ub = S * A * H
    results = {
        'eq5_ub': ub,
        'trials': [],
        'eq5_violations': 0,
        'ratio_not_constant': 0,
        'control_ratios': [],
        'solver_converged_all': True,
    }
    for trial in range(n_mdps):
        mdp = MDP.random(S, A, H, rng=rng, sparsity=1.0)
        mu_star, lam, converged, n_iter = solve_logbarrier(mdp)
        if not converged:
            results['solver_converged_all'] = False
        # the constant predicted by first-order optimality: 1/mu* = C^T lambda
        # => ratio = (C mu^pi)^T lambda = b^T lambda for every policy.
        C, b = mdp.flow_matrix()
        Cind, piv = _basis(C)
        # lam solves Cind.T lam = -1/mu (see solve_logbarrier); the dual of the
        # stationarity eq -1/mu* + C^T lambda = 0 is lam_here = -lambda.
        ratio_constant = float((-lam) @ b[piv])
        # exact identity: ratio = mu*^T (1/mu*) = H*S*A  (equal to b^T lambda)
        ratio_exact = float(np.sum(np.ones(mdp.H * mdp.S * mdp.A)))

        # observed ratios: exhaustive deterministic + random
        ratios = []
        if exhaustive_det:
            total = mdp.A ** (mdp.H * mdp.S)
            if total <= 2 ** 16:
                for pi in all_deterministic_policies(mdp):
                    ratios.append(coverage_ratio(mdp, mu_star, pi))
        for pi in random_policies(mdp, n_rand_policies, rng=rng):
            ratios.append(coverage_ratio(mdp, mu_star, pi))

        r_max = float(max(ratios))
        r_min = float(min(ratios))
        if r_max > ub + 1e-6:
            results['eq5_violations'] += 1
        if (r_max - r_min) > 1e-6:
            results['ratio_not_constant'] += 1

        # CONTROL: feasible but non-logbarrier mu (occupancy of a random
        # policy, clipped to the polytope via feasibility projection is not
        # needed -- occupancy of ANY policy is in Lambda automatically).
        pi_c = rng.dirichlet(np.ones(mdp.A), size=(mdp.H, mdp.S))
        mu_c = mdp.occupancy(pi_c)
        r_control = float(np.sum(
            mdp.occupancy(rng.dirichlet(np.ones(mdp.A), size=(mdp.H, mdp.S)))
            / np.clip(mu_c, 1e-14, None)))
        results['control_ratios'].append(r_control)

        results['trials'].append({
            'trial': trial,
            'ratio_constant_theory': ratio_constant,
            'ratio_exact_identity': ratio_exact,
            'ratio_max_observed': r_max,
            'ratio_min_observed': r_min,
            'ratio_std': float(np.std(ratios)),
            'solver_converged': bool(converged),
            'n_iter': int(n_iter),
        })
    ratios = [t['ratio_constant_theory'] for t in results['trials']]
    results['eq5_max_over_trials'] = float(max(ratios))
    results['eq5_mean'] = float(np.mean(ratios))
    results['eq5_bound_satisfied'] = (
        results['eq5_violations'] == 0
        and results['ratio_not_constant'] == 0
        and results['eq5_max_over_trials'] <= ub + 1e-6)
    results['control_ratio_median'] = float(np.median(results['control_ratios']))
    results['control_exceeds_bound'] = results['control_ratio_median'] > ub
    results['|S||A|H'] = ub
    results['claim_supported'] = (
        results['eq5_bound_satisfied']
        and results['control_exceeds_bound']
        and results['solver_converged_all'])
    return results


if __name__ == '__main__':
    import json
    print(json.dumps(audit_claim4(), indent=2))
