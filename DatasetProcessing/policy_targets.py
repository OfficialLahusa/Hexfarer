"""Policy targets from a search's root, per position over its legal moves.

visit_targets: the visit shares, optionally with a temperature (visits^(1/T), what the self-play loops train on with T 2).
gumbel_targets: the improved policy of Gumbel MuZero (Danihelka et al., 2022, "Policy improvement by planning with
Gumbel"): softmax(log prior + sigma(completed Q)), with sigma(q) = (c_visit + most visits) * c_scale * q. It moves away
from the prior only as far as the search's values separate the moves, and more so the more the search visited.
completed_q: the values of the visited moves, and for the unvisited ones the mixed value v_mix of the paper (the root
value mixed with the prior-weighted mean of the visited moves' values).
"""

import numpy as np

# Floor of a prior before its logarithm, so a move the net rules out stays (almost) ruled out
PRIOR_FLOOR = 1e-30


def visit_targets(visits, temperature: float = 1.0) -> np.ndarray:
    """The visit shares, visits^(1/temperature) normalized; ValueError without visits."""
    visits = np.asarray(visits, dtype=np.float64)
    total = visits.sum()
    if total <= 0:
        raise ValueError("A policy target needs visits.")
    shares = visits / total
    if temperature != 1.0:
        shares = shares ** (1.0 / temperature)
        shares = shares / shares.sum()
    return shares


def completed_q(prior, visits, q, root_value=None) -> np.ndarray:
    """The visited moves' values as they are, the unvisited ones' replaced by v_mix:
    (root value + N / (sum of the visited moves' priors) * sum over the visited moves of prior * q) / (1 + N), N the total
    visits. Without a root value the prior-weighted mean of the visited values stands in for it."""
    prior = np.asarray(prior, dtype=np.float64)
    visits = np.asarray(visits, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    visited = visits > 0
    if not visited.any():
        raise ValueError("Completed values need a visited move.")
    weight = prior[visited].sum()
    # Priors of the visited moves can all be 0 (a move the net rules out, searched anyway): then weigh them equally
    if weight <= 0:
        mean = q[visited].mean()
    else:
        mean = float(prior[visited] @ q[visited]) / weight
    if root_value is None:
        root_value = mean
    total = visits.sum()
    v_mix = (root_value + total * mean) / (1 + total)
    completed = q.copy()
    completed[~visited] = v_mix
    return completed


def gumbel_targets(prior, visits, q, c_visit: float = 50.0, c_scale: float = 0.1, value_range=None, root_value=None) -> np.ndarray:
    """softmax(log prior + (c_visit + most visits) * c_scale * completed Q). With value_range (min, max) the values are
    normalized to it first (clipped to [0, 1]), without it they stay raw win probabilities. A single visited move (the
    search's shortcut for a winning move) gets the whole target, where the prior alone would teach nothing."""
    prior = np.asarray(prior, dtype=np.float64)
    visits = np.asarray(visits, dtype=np.float64)
    visited = visits > 0
    if visited.sum() == 1:
        return visited.astype(np.float64)
    completed = completed_q(prior, visits, q, root_value)
    if value_range is not None:
        low, high = value_range
        completed = np.clip((completed - low) / max(high - low, 1e-12), 0.0, 1.0)
    logits = np.log(np.maximum(prior, PRIOR_FLOOR)) + (c_visit + visits.max()) * c_scale * completed
    logits -= logits.max()
    target = np.exp(logits)
    return target / target.sum()
