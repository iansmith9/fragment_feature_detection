"""Curve fitting helpers.

- `fit_gaussian_elution`: describe an NMF elution profile (a column of W) by a Gaussian
  apex `mu` and width `sigma` in retention time.
- `approximate_overlap_curves`: how much two fitted elution peaks overlap, used to
  find the same component in overlapping scan windows.
- `least_squares_with_l1_bounds`: the sparse, bounded regression that links an MS2
  component's elution profile to MS1 feature profiles.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit, minimize


def approximate_overlap_curves(
    m1: float,
    s1: float,
    m2: float,
    s2: float,
    bounds: Tuple[int, int],
) -> np.ndarray:
    """Calculates the overlap between two Gaussian curves.

    Computes the overlap between two Gaussian probability density functions by finding
    the minimum value at each point and integrating. The result is the fraction of
    curve 1's area that lies under both curves: 1 for identical curves, near 0 for
    well-separated ones. It isn't symmetric when the widths differ.

    The integral is a sum over 10,000 evenly spaced points, so `bounds` should cover
    both curves (a few sigma either side) without being much wider.

    Args:
        m1 (float): Mean of first Gaussian
        s1 (float): Standard deviation of first Gaussian
        m2 (float): Mean of second Gaussian
        s2 (float): Standard deviation of second Gaussian
        bounds (Tuple[int, int]): Integration bounds (min, max)

    Returns:
        np.ndarray: Overlap ratio between the two curves
    """
    x = np.linspace(*bounds, 10000)
    p1 = stats.norm(m1, s1).pdf(x)
    p2 = stats.norm(m2, s2).pdf(x)

    # Area under min(p1, p2), relative to the area under p1 (same grid spacing).
    p_min = np.min(np.vstack([p1, p2]), axis=0)

    return p_min.sum() / p1.sum()


def gaussian_curve(
    x: np.ndarray,
    x0: float,
    s0: float,
) -> np.ndarray:
    """Computes unit-scaled Gaussian curve.

    exp(-(x - x0)^2 / (2 s0^2)), with height 1 at x0 (not a normalized density).

    Args:
        x (np.ndarray): Input x values
        x0 (float): Mean of Gaussian
        s0 (float): Standard deviation of Gaussian

    Returns:
        (np.ndarray): Gaussian curve values
    """
    return np.exp(-((x - x0) ** 2) / (2 * s0**2))


def fit_gaussian_elution(
    t: np.ndarray,
    m: np.ndarray,
    mu_bounds: Optional[Tuple[float, float]] = None,
    sigma_bounds: Optional[Tuple[float, float]] = (0.1, 10.0),
    clip_width: Optional[int] = None,
    **kwargs: Dict[str, Any],
) -> np.ndarray:
    """Fits Gaussian shape to NMF peak elutions.

    The profile is min-max scaled to [0, 1] and fitted with a unit-height Gaussian, so
    only the apex (mu) and width (sigma) are free. The starting guess is the
    intensity-weighted mean and standard deviation of the profile (its first two
    moments). The returned error is the mean absolute difference between the scaled
    profile and the fitted curve (0 is a perfect fit, at most 1).

    Args:
        t (np.ndarray): Retention time axis
        m (np.ndarray): Input elution profile
        mu_bounds (Tuple[float, float]): Bounds for mean parameter
        sigma_bounds (Tuple[float, float]): Bounds for standard deviation
        clip_width (int): Number of points to drop at each end before fitting
        **kwargs (Dict[str, Any]): Ignored (lets callers pass a whole config section).

    Returns:
        popt (List[float]): [mean, std] parameters
        pcov (np.ndarray): Covariance matrix
        error (float): Fit error (mean absolute error of the scaled profile)

    Raises:
        ValueError: If the profile is empty or constant.
        RuntimeError: If curve_fit doesn't converge.
    """
    m_scale = m.copy()

    if clip_width:
        m_scale = m_scale[clip_width : -1 * clip_width]
        t = t[clip_width : -1 * clip_width]

    if m_scale.shape[0] == 0 or m_scale.max() == m_scale.min():
        raise ValueError("Elution profile is empty or constant, can't fit gaussian.")
    # Scale to [0, 1] to match the unit-height gaussian_curve.
    m_scale = (m_scale - m_scale.min()) / (m_scale.max() - m_scale.min())

    # Moment estimates for the starting point: weighted mean and standard deviation.
    m0 = sum(t * m_scale) / sum(m_scale)
    # m0 = sum(m_scale[:, 0] * m_scale[:, 1]) / sum(m_scale[:, 1])
    s0 = np.sqrt(sum(m_scale * (t - m0) ** 2) / sum(m_scale))
    # s0 = np.sqrt(sum(m_scale[:, 1] * (m_scale[:, 0] - m0) ** 2) / sum(m_scale[:, 1]))

    # curve_fit bounds as [[mu_low, sigma_low], [mu_high, sigma_high]].
    bounds = [[-np.inf, -np.inf], [np.inf, np.inf]]
    if mu_bounds:
        bounds[0][0] = mu_bounds[0]
        bounds[1][0] = mu_bounds[1]
    if sigma_bounds:
        bounds[0][1] = sigma_bounds[0]
        bounds[1][1] = sigma_bounds[1]

    # curve_fit needs a feasible starting point. A guess outside its bounds is moved
    # to the nearest bound (e.g. a sharp peak starts at the lower sigma bound, not at
    # the upper one).
    if mu_bounds:
        m0 = float(np.clip(m0, mu_bounds[0], mu_bounds[1]))
    if sigma_bounds:
        s0 = float(np.clip(s0, sigma_bounds[0], sigma_bounds[1]))

    # Bounded nonlinear least squares (trust region reflective).
    popt, pcov = curve_fit(
        gaussian_curve,
        t,
        m_scale,
        p0=[m0, s0],
        bounds=bounds,
    )

    # Error against the fitted unit-height curve, not the density-normalized pdf.
    e = np.absolute(gaussian_curve(t, *popt) - m_scale).mean()

    return popt, pcov, e


def least_squares_with_l1_bounds(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Performs L1-regularized least squares optimization with bounds.

    Minimizes the objective function:
    sum((X @ coef - y)^2) + alpha * sum(|coef|)
    subject to the given bounds constraints on coefficients.

    In MS1 matching, the columns of X are unit-norm MS1 feature elution profiles and
    y is the unit-norm MS2 component profile, so a coefficient near 1 means one MS1
    feature explains the component on its own. The L1 term pushes most coefficients
    to exactly 0. Solved with SLSQP (numerical gradients, at most 100 iterations),
    starting from all zeros.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        alpha (float): L1 regularization strength. Defaults to 0.1
        bounds (Tuple[float, float]): (min, max) bounds for coefficients.
            Defaults to (0.0, 1.0)

    Returns:
        np.ndarray: Optimized coefficient vector
    """

    def objective(coef: np.ndarray) -> float:
        """Squared residual plus L1 penalty for a coefficient vector."""
        return np.sum((np.dot(X, coef) - y) ** 2) + alpha * np.sum(np.abs(coef))

    coef_init = np.zeros(X.shape[1])

    # Same (min, max) box for every coefficient.
    coef_bounds = [bounds for _ in coef_init]

    result = minimize(
        objective,
        coef_init,
        method="SLSQP",
        bounds=coef_bounds,
        options={"maxiter": 100},
    )

    return result.x
