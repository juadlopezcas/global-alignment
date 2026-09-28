"""Controlled best-improvement sign search with one MMD solve on SO(d).

Rows are points: Y = X @ R_true.T + noise and X_aligned = X @ R_est.T.
All sources have theoretical covariance diag(d, d-1, ..., 1). The existing
reported frame error compares R_true @ U_X with U_Y, allowing signed axis
permutations; rotation error compares R_true with R_est without that equivalence.
Ground truth is passed only to evaluation, never to registration.

Run all three datasets (300 trials with the defaults)::

    python numerical_experiments.py

For a small sanity run::

    python numerical_experiments.py --dimensions 2 3 5 --num-trials 2

To run one dataset::

    python numerical_experiments.py --dataset-types uniform_box --num-trials 2

There is one final MMD optimization, initialized only by best-improvement
pairwise greedy sign refinement. The two PCA optimizations are unchanged.
First-improvement instead accepts the first improving pair; this script scores
all pairs and accepts the best. Both use the same stopping convention and
remain local heuristics. Results default to numerical_results/best_improvement_controlled;
previous results and numerical_experiments_first_improvement.py are unchanged.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import autograd.numpy as anp
import numpy as np
from pymanopt import Problem
from pymanopt.function import autograd
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import TrustRegions
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


DATASET_TYPES = ("gaussian_ellipsoid", "uniform_box", "laplace_product")
DATASET_LABELS = {
    "gaussian_ellipsoid": "Gaussian ellipsoid",
    "uniform_box": "Uniform box",
    "laplace_product": "Laplace product",
}
# Fixed identifiers, independent of CLI order and Python's randomized hash().
DATASET_SEED_IDS = {"gaussian_ellipsoid": 0, "uniform_box": 1, "laplace_product": 2}


@dataclass
class GreedyResult:
    signs: np.ndarray
    history: list[float]
    accepted_pairs: list[tuple[int, int]]
    runtime_seconds: float
    num_pair_evaluations: int
    converged: bool

    @property
    def num_accepted_moves(self) -> int:
        return len(self.accepted_pairs)

    @property
    def stopping_criterion(self) -> str:
        return "pairwise_local_optimum" if self.converged else "max_greedy_moves"


@dataclass
class OptimizationResult:
    label: str
    R: np.ndarray
    initial_mmd: float
    final_mmd: float
    iterations: int
    gradient_norm: float
    stopping_criterion: str


@dataclass
class PCAResult:
    frame: np.ndarray
    iterations: int
    gradient_norm: float
    stopping_criterion: str


@dataclass
class RegistrationResult:
    R_est: np.ndarray
    U_X: np.ndarray
    U_Y: np.ndarray
    final_mmd: float
    optimization: OptimizationResult
    greedy: GreedyResult
    R_init: np.ndarray
    pca_X: PCAResult
    pca_Y: PCAResult


@dataclass(frozen=True)
class ExperimentConfig:
    dimensions: tuple[int, ...] = (2, 3, 5, 10, 20)
    n: int = 200
    num_trials: int = 20
    seed: int = 42
    noise_std: float = 0.05
    sigma: float = 1.0
    max_greedy_moves: int = 100
    greedy_tolerance: float = 1e-12
    max_iterations: int = 100
    pca_max_iterations: int = 200
    min_gradient_norm: float = 1e-6
    dataset_types: tuple[str, ...] = DATASET_TYPES

    def validate(self) -> None:
        if not self.dataset_types or any(name not in DATASET_TYPES for name in self.dataset_types):
            raise ValueError(f"dataset_types must contain names from {DATASET_TYPES}.")
        if len(set(self.dataset_types)) != len(self.dataset_types):
            raise ValueError("Dataset types must be unique.")
        if not self.dimensions or any(d < 2 for d in self.dimensions):
            raise ValueError("Every dimension must be at least 2.")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Dimensions must be unique.")
        if self.n < 2 or self.num_trials < 1:
            raise ValueError("Require n >= 2 and num_trials >= 1.")
        if self.seed < 0 or self.max_greedy_moves < 0:
            raise ValueError("Seed and max_greedy_moves must be nonnegative.")
        if not np.isfinite(self.greedy_tolerance) or self.greedy_tolerance < 0:
            raise ValueError("greedy_tolerance must be finite and nonnegative.")
        if self.max_iterations < 1 or self.pca_max_iterations < 1:
            raise ValueError("Optimizer iteration limits must be positive.")
        if not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative.")
        _validate_sigma(self.sigma)
        if not np.isfinite(self.min_gradient_norm) or self.min_gradient_norm <= 0:
            raise ValueError("min_gradient_norm must be finite and positive.")


def assert_rotation(R: np.ndarray, atol: float = 1e-6) -> None:
    """Reject invalid SO(d) matrices, including reflections."""
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or not np.isfinite(R).all():
        raise ValueError("Rotation must be a finite square matrix.")
    if not np.allclose(R.T @ R, np.eye(R.shape[0]), atol=atol, rtol=0):
        raise ValueError("Rotation is not orthogonal.")
    if not np.isclose(np.linalg.det(R), 1.0, atol=atol, rtol=0):
        raise ValueError("Rotation must have determinant +1.")


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian QR with diagonal-sign and determinant correction to SO(d)."""
    if d < 2:
        raise ValueError("Registration dimension must be at least 2.")
    Q, triangular = np.linalg.qr(rng.standard_normal((d, d)))
    Q *= np.where(np.diag(triangular) < 0, -1.0, 1.0)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    assert_rotation(Q)
    return Q


def generate_ellipsoid(
    n: int, d: int, rng: np.random.Generator, noise_std: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible Gaussian wrapper with the original RNG draw order."""
    return generate_dataset(n, d, "gaussian_ellipsoid", rng, noise_std)


def generate_source(
    n: int, d: int, dataset_type: str, rng: np.random.Generator
) -> np.ndarray:
    """Independent coordinates with common theoretical covariance diag(d,...,1).

    Shape changes while population principal variances remain controlled.
    Finite samples need not have exactly this covariance. Shared radius
    normalization is applied later, unchanged, by the registration experiment.
    """
    if n < 2 or d < 2:
        raise ValueError("Require n,d >= 2.")
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"Unknown dataset_type {dataset_type!r}; choose from {DATASET_TYPES}.")
    axis_std = np.sqrt(np.arange(d, 0, -1, dtype=float))
    if dataset_type == "gaussian_ellipsoid":
        return rng.standard_normal((n, d)) * axis_std
    if dataset_type == "uniform_box":
        # Var(Uniform(-c,c)) = c^2/3, so c = sqrt(3) * axis_std.
        half_width = np.sqrt(3) * axis_std
        return rng.uniform(-half_width, half_width, size=(n, d))
    # Var(Laplace(0,b)) = 2*b^2, so b = axis_std/sqrt(2).
    return rng.laplace(0.0, axis_std / np.sqrt(2), size=(n, d))


def generate_dataset(
    n: int, d: int, dataset_type: str, rng: np.random.Generator,
    noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Common observation model Y = X @ R_true.T + isotropic Gaussian noise."""
    if n < 2 or d < 2 or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Require n,d >= 2 and finite noise_std >= 0.")
    X = generate_source(n, d, dataset_type, rng)
    R_true = random_rotation(d, rng)
    Y = X @ R_true.T + noise_std * rng.standard_normal((n, d))
    return X, Y, R_true


def validate_generators(n: int = 100_000, d: int = 5) -> list[dict]:
    """One temporary source sample per distribution, never inside a trial.

    Check standardized means, coordinate variances, cross-covariances and
    marginal Pearson kurtosis. Theoretical kurtoses (3, 1.8, 6) distinguish the
    sampling laws without introducing any new registration/evaluation metric.
    These are deterministic sanity checks, not confidence-interval estimates.
    """
    if n < 10_000 or d < 2:
        raise ValueError("Generator validation requires n >= 10000 and d >= 2.")
    axis_std = np.sqrt(np.arange(d, 0, -1, dtype=float))
    kurtosis_bounds = {"gaussian_ellipsoid": (2.7, 3.3), "uniform_box": (1.7, 1.9),
                       "laplace_product": (5.0, 7.0)}
    diagnostics = []
    for dataset_type in DATASET_TYPES:
        rng = np.random.default_rng(np.random.SeedSequence([20260917, DATASET_SEED_IDS[dataset_type]]))
        standardized = generate_source(n, d, dataset_type, rng) / axis_std
        centered = standardized - standardized.mean(axis=0)
        covariance = centered.T @ centered / (n - 1)
        mean_error = float(np.abs(standardized.mean(axis=0)).max())
        variance_error = float(np.abs(np.diag(covariance) - 1).max())
        cross_covariance = covariance - np.diag(np.diag(covariance))
        cross_error = float(np.abs(cross_covariance).max())
        kurtosis = np.mean(centered**4, axis=0) / np.mean(centered**2, axis=0)**2
        lower, upper = kurtosis_bounds[dataset_type]
        if mean_error > 0.02 or variance_error > 0.05 or cross_error > 0.03:
            raise AssertionError(f"{dataset_type}: source mean/covariance validation failed.")
        if not np.all((kurtosis > lower) & (kurtosis < upper)):
            raise AssertionError(f"{dataset_type}: marginal shape validation failed: {kurtosis}.")
        if dataset_type == "uniform_box" and np.abs(standardized).max() > np.sqrt(3):
            raise AssertionError("Uniform source exceeds its specified support.")
        diagnostics.append({
            "dataset_type": dataset_type, "sample_size": n, "dimension": d,
            "max_standardized_mean_error": mean_error,
            "max_relative_variance_error": variance_error,
            "max_standardized_cross_covariance": cross_error,
            "mean_marginal_kurtosis": float(kurtosis.mean()),
        })
    return diagnostics


def trial_rngs(
    seed: int, dimension: int, trial: int, dataset_type: str
) -> tuple[np.random.Generator, np.random.Generator]:
    """Stable dataset/dimension/trial streams, preserving legacy Gaussian draws."""
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"Unknown dataset_type {dataset_type!r}.")
    entropy = [seed, dimension, trial]
    if dataset_type != "gaussian_ellipsoid":
        entropy.append(DATASET_SEED_IDS[dataset_type])
    data_seed, fit_seed = np.random.SeedSequence(entropy).spawn(2)
    return np.random.default_rng(data_seed), np.random.default_rng(fit_seed)


def _validate_clouds(X: np.ndarray, Y: np.ndarray) -> None:
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("Clouds must be 2D arrays with equal coordinate dimension.")
    if min(len(X), len(Y)) < 2 or X.shape[1] < 2:
        raise ValueError("Each cloud needs at least two points in dimension >= 2.")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Clouds must have finite coordinates.")


def _validate_sigma(sigma: float) -> None:
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Kernel bandwidth sigma must be finite and positive.")


def center_and_normalize(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Remove centroids and use one common radius, as in 2d.py."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    _validate_clouds(X, Y)
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    scale = float(max(np.linalg.norm(Xc, axis=1).max(), np.linalg.norm(Yc, axis=1).max()))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The combined centered radius must be positive and finite.")
    return Xc / scale, Yc / scale, scale


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    _validate_sigma(sigma)
    return np.exp(-cdist(X, Y, metric="sqeuclidean") / (2 * sigma**2))


def mmd_squared(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """Diagonal-excluding MMD^2 formula used by 2d.py; may be negative.

    This is often called the unbiased two-sample estimator. Paired synthetic
    X/Y observations are dependent, so here it is used as a registration loss,
    without claiming an unbiased population estimate under that sampling model.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    _validate_clouds(X, Y)
    Kxx, Kyy, Kxy = rbf_kernel(X, X, sigma), rbf_kernel(Y, Y, sigma), rbf_kernel(X, Y, sigma)
    n, m = len(X), len(Y)
    return float(
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2 * Kxy.mean()
    )


class RotationMMD:
    """Cache within-cloud terms, which are constant on SO(d).

    The cross term retains the norms of the rotated coordinates in the Autograd
    expression. The resulting smooth objective equals full MMD on the manifold;
    Pymanopt supplies its Riemannian gradient and Hessian.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, sigma: float = 1.0):
        self.X, self.Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        _validate_clouds(self.X, self.Y)
        _validate_sigma(sigma)
        self.sigma = sigma
        Kxx = rbf_kernel(self.X, self.X, sigma)
        Kyy = rbf_kernel(self.Y, self.Y, sigma)
        n, m = len(self.X), len(self.Y)
        self.within = float(
            (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        )

    def evaluate(self, R: np.ndarray) -> float:
        return float(self.within - 2 * rbf_kernel(self.X @ R.T, self.Y, self.sigma).mean())

    def problem(self, manifold: SpecialOrthogonalGroup) -> Problem:
        X, Y, sigma, within = self.X, self.Y, self.sigma, self.within
        Y_sq = anp.sum(Y**2, axis=1)[None, :]

        @autograd(manifold)
        def cost(R):
            X_rotated = anp.dot(R, X.T).T
            squared_distances = (
                anp.sum(X_rotated**2, axis=1)[:, None]
                + Y_sq
                - 2 * anp.dot(X_rotated, Y.T)
            )
            return within - 2 * anp.mean(anp.exp(-squared_distances / (2 * sigma**2)))

        return Problem(manifold=manifold, cost=cost)


def estimate_pca_frame(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    max_iterations: int = 200,
    min_gradient_norm: float = 1e-6,
) -> PCAResult:
    """Ordered geometric frame from weighted-variance optimization on SO(d)."""
    _validate_clouds(data, data)
    d = data.shape[1]
    C = data.T @ data
    W = anp.diag(anp.arange(d, 0, -1, dtype=float))
    manifold = SpecialOrthogonalGroup(d)

    @autograd(manifold)
    def pca_cost(U):
        return -anp.trace(W @ U.T @ C @ U)

    problem = Problem(manifold=manifold, cost=pca_cost)
    optimizer = TrustRegions(
        max_iterations=max_iterations, min_gradient_norm=min_gradient_norm,
        use_rand=False, verbosity=0,
    )
    # Explicit Generator-supplied start avoids Pymanopt's global np.random state.
    result = optimizer.run(problem, initial_point=random_rotation(d, rng))
    assert_rotation(result.point)
    return PCAResult(result.point, result.iterations, float(result.gradient_norm), result.stopping_criterion)


def greedy_sign_refinement(
    Xc: np.ndarray,
    Yc: np.ndarray,
    U_X: np.ndarray,
    U_Y: np.ndarray,
    sigma: float = 1.0,
    max_greedy_moves: int = 100,
    tolerance: float = 1e-12,
) -> GreedyResult:
    """Best-improvement pairwise greedy; one move is one accepted pair flip.

    Scan ALL pairs (i,j), i<j, in lexicographic order, then accept the lowest
    candidate MMD among pairs improving the current score by more than tolerance.
    Restart a complete scan after every accepted move. First-improvement differs
    only by immediately accepting the first improving pair and restarting.
    Both search all determinant-preserving two-axis flips, with no fixed pivot
    or parity axis. They can take different paths to different pairwise local
    minima; neither guarantees the global optimum over the 2^(d-1) sign patterns.

    Normal termination requires a COMPLETE scan with no improvement. The move
    limit is only a safety cap; reaching it does not certify local optimality.
    A pair flip updates the cached PCA-coordinate cross term by two rank-one
    terms. No TrustRegions solve occurs here. Pair counts exclude the initial
    score and include every evaluated proposal, including the final full scan.
    Timing includes setup, initial scoring, and all candidate scoring.
    """
    start_time = perf_counter()
    if max_greedy_moves < 0:
        raise ValueError("max_greedy_moves must be nonnegative.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative.")
    assert_rotation(U_X)
    assert_rotation(U_Y)
    objective = RotationMMD(Xc, Yc, sigma)
    d = Xc.shape[1]
    X_axes, Y_axes = Xc @ U_X, Yc @ U_Y
    signs = np.ones(d, dtype=int)
    cross = X_axes @ Y_axes.T
    norm_sum = np.sum(X_axes**2, axis=1)[:, None] + np.sum(Y_axes**2, axis=1)[None, :]

    def score(inner_products):
        return float(objective.within - 2 * np.exp(-(norm_sum - 2 * inner_products) / (2 * sigma**2)).mean())

    current = score(cross)
    history = [current]
    accepted_pairs = []
    num_pair_evaluations = 0
    converged = False
    for _ in range(max_greedy_moves):
        best_score, best_pair, best_cross = current, None, None
        for i in range(d - 1):
            flip_i = 2 * signs[i] * np.outer(X_axes[:, i], Y_axes[:, i])
            for j in range(i + 1, d):
                candidate_cross = cross - flip_i - 2 * signs[j] * np.outer(X_axes[:, j], Y_axes[:, j])
                candidate_score = score(candidate_cross)
                num_pair_evaluations += 1
                if candidate_score < current - tolerance and candidate_score < best_score:
                    best_score, best_pair, best_cross = candidate_score, (i, j), candidate_cross
        # No early acceptance: every proposed pair above is scored before this.
        if best_pair is None:
            converged = True
            break
        # Starting at all +1 and flipping two signs keeps product +1 for every
        # proposed and accepted pattern, not just the final one.
        signs[list(best_pair)] *= -1
        cross, current = best_cross, best_score
        history.append(current)
        accepted_pairs.append(best_pair)
    return GreedyResult(signs, history, accepted_pairs, perf_counter() - start_time,
                        num_pair_evaluations, converged)


def register_point_clouds(
    Xc: np.ndarray,
    Yc: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float = 1.0,
    max_greedy_moves: int = 100,
    greedy_tolerance: float = 1e-12,
    max_iterations: int = 100,
    pca_max_iterations: int = 200,
    min_gradient_norm: float = 1e-6,
) -> RegistrationResult:
    """Two unchanged PCA solves, then ONE MMD solve from the greedy rotation."""
    Xc, Yc = np.asarray(Xc, dtype=float), np.asarray(Yc, dtype=float)
    _validate_clouds(Xc, Yc)
    pca_X = estimate_pca_frame(Xc, rng, max_iterations=pca_max_iterations, min_gradient_norm=min_gradient_norm)
    pca_Y = estimate_pca_frame(Yc, rng, max_iterations=pca_max_iterations, min_gradient_norm=min_gradient_norm)
    greedy = greedy_sign_refinement(
        Xc, Yc, pca_X.frame, pca_Y.frame, sigma, max_greedy_moves, greedy_tolerance,
    )
    R_init = (pca_Y.frame * greedy.signs) @ pca_X.frame.T
    assert_rotation(R_init)
    objective = RotationMMD(Xc, Yc, sigma)
    problem = objective.problem(SpecialOrthogonalGroup(Xc.shape[1]))
    initial_mmd = objective.evaluate(R_init)
    optimizer = TrustRegions(
        max_iterations=max_iterations, min_gradient_norm=min_gradient_norm,
        use_rand=False, verbosity=0,
    )
    result = optimizer.run(problem, initial_point=R_init)
    assert_rotation(result.point)
    final_mmd = objective.evaluate(result.point)
    if not np.isfinite(final_mmd):
        raise RuntimeError("Nonfinite final MMD for greedy_best_improvement.")
    optimization = OptimizationResult(
        "greedy_best_improvement", result.point, initial_mmd, final_mmd,
        result.iterations, float(result.gradient_norm), result.stopping_criterion,
    )
    return RegistrationResult(
        result.point, pca_X.frame, pca_Y.frame, final_mmd, optimization,
        greedy, R_init, pca_X, pca_Y,
    )


def signed_permutation_frame_error(
    F_true: np.ndarray, F_est: np.ndarray
) -> tuple[float, float]:
    """min_S ||F_true - F_est @ S||_F for all signed permutations S.

    The d x d assignment matches frame axes ONLY, not the n observed points.
    Signs are unrestricted here (including determinant -1); frame equivalence
    is intentionally broader than equivalence of physical registration rotations.
    """
    F_true, F_est = np.asarray(F_true, dtype=float), np.asarray(F_est, dtype=float)
    if F_true.ndim != 2 or F_true.shape != F_est.shape or F_true.shape[0] != F_true.shape[1] or not len(F_true):
        raise ValueError("Frames must be square matrices of the same nonzero size.")
    d = F_true.shape[0]
    for frame in (F_true, F_est):
        if not np.isfinite(frame).all() or not np.allclose(frame.T @ frame, np.eye(d), atol=1e-6, rtol=0):
            raise ValueError("Frames must be finite orthonormal bases.")
    C = F_true.T @ F_est
    true_axes, estimated_axes = linear_sum_assignment(-np.abs(C))
    signs = np.where(C[true_axes, estimated_axes] < 0, -1.0, 1.0)
    matched_frame = np.empty_like(F_est)
    matched_frame[:, true_axes] = F_est[:, estimated_axes] * signs
    frame_error = float(np.linalg.norm(F_true - matched_frame, ord="fro"))
    return frame_error, frame_error / np.sqrt(d)


def validate_frame_error() -> None:
    """Fast required metric checks, automatically run before any experiment."""
    rng = np.random.default_rng(2026)
    F = random_rotation(5, rng)
    permutation = [2, 4, 1, 0, 3]
    signs = np.array([-1, 1, -1, -1, 1])
    for equivalent in (F, F[:, permutation], F * signs, F[:, permutation] * signs):
        np.testing.assert_allclose(signed_permutation_frame_error(F, equivalent), (0.0, 0.0), atol=1e-12)
    angle = 0.31
    R1 = np.eye(2)
    different = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    raw, normalized = signed_permutation_frame_error(R1, different)
    if raw <= 0 or not np.isclose(normalized, raw / np.sqrt(2)):
        raise AssertionError("A genuinely different frame must have positive error.")
    R2 = -np.eye(2)  # Both matrices are in SO(2), but are different rotations.
    assert_rotation(R1)
    assert_rotation(R2)
    np.testing.assert_allclose(signed_permutation_frame_error(R1, R2), (0.0, 0.0), atol=1e-12)
    if np.linalg.norm(R1 - R2, ord="fro") <= 0:
        raise AssertionError("Frame equivalence must not erase rotation error.")


def run_trial(
    config: ExperimentConfig, dimension: int, trial: int,
    dataset_type: str = "gaussian_ellipsoid",
) -> dict:
    """Independent, order-stable trial streams; ground truth used only below fit."""
    data_rng, fit_rng = trial_rngs(config.seed, dimension, trial, dataset_type)
    start_time = perf_counter()
    X, Y, R_true = generate_dataset(config.n, dimension, dataset_type, data_rng, config.noise_std)
    Xc, Yc, scale = center_and_normalize(X, Y)
    result = register_point_clouds(
        Xc, Yc, fit_rng, sigma=config.sigma,
        max_greedy_moves=config.max_greedy_moves, greedy_tolerance=config.greedy_tolerance,
        max_iterations=config.max_iterations,
        pca_max_iterations=config.pca_max_iterations, min_gradient_norm=config.min_gradient_norm,
    )
    # Evaluation boundary: the fitting functions above never receive R_true.
    # frame_error, normalized_frame_error = signed_permutation_frame_error(R_true, result.U_Y)

    F_Y_true_from_X = R_true @ result.U_X

    frame_error, normalized_frame_error = (
        signed_permutation_frame_error(
            F_Y_true_from_X,
            result.U_Y
        )
    )

    rotation_error = float(np.linalg.norm(R_true - result.R_est, ord="fro"))
    normalized_rotation_error = rotation_error / np.sqrt(dimension)
    winner = result.optimization
    return {
        "dataset_type": dataset_type,
        "dimension": dimension, "trial": trial, "seed": config.seed,
        "n": config.n, "noise_std": config.noise_std, "sigma": config.sigma,
        # Fixed compatibility fields: this variant always requests/runs one start.
        "requested_starts": 1, "actual_starts": 1,
        "max_greedy_moves": config.max_greedy_moves, "greedy_tolerance": config.greedy_tolerance,
        "max_iterations": config.max_iterations,
        "pca_max_iterations": config.pca_max_iterations, "min_gradient_norm": config.min_gradient_norm,
        "normalization_scale": scale, "frame_error": frame_error,
        "normalized_frame_error": normalized_frame_error, "rotation_error": rotation_error,
        "normalized_rotation_error": normalized_rotation_error,
        "final_mmd": result.final_mmd, "runtime_seconds": perf_counter() - start_time,
        "greedy_runtime_seconds": result.greedy.runtime_seconds,
        "greedy_num_accepted_moves": result.greedy.num_accepted_moves,
        "greedy_num_pair_evaluations": result.greedy.num_pair_evaluations,
        "greedy_converged": result.greedy.converged,
        "greedy_stopping_criterion": result.greedy.stopping_criterion,
        "selected_start": winner.label, "winner_iterations": winner.iterations,
        "winner_gradient_norm": winner.gradient_norm, "winner_stopping_criterion": winner.stopping_criterion,
        "unconverged_starts": int(winner.gradient_norm > config.min_gradient_norm),
        "pca_X_gradient_norm": result.pca_X.gradient_norm,
        "pca_Y_gradient_norm": result.pca_Y.gradient_norm,
        "pca_X_stopping_criterion": result.pca_X.stopping_criterion,
        "pca_Y_stopping_criterion": result.pca_Y.stopping_criterion,
    }


METRICS = ("frame_error", "normalized_frame_error", "rotation_error", "normalized_rotation_error", "final_mmd", "runtime_seconds")
METRICS += ("greedy_runtime_seconds", "greedy_num_accepted_moves", "greedy_num_pair_evaluations")


def aggregate_results(rows: list[dict]) -> list[dict]:
    """Group by dataset AND dimension; sample SD is undefined for one trial."""
    summaries = []
    groups = {(row["dataset_type"], row["dimension"]) for row in rows}
    for dataset_type, dimension in sorted(groups, key=lambda group: (DATASET_SEED_IDS[group[0]], group[1])):
        trials = [row for row in rows if (row["dataset_type"], row["dimension"]) == (dataset_type, dimension)]
        summary = {"dataset_type": dataset_type, "dimension": dimension, "num_trials": len(trials)}
        for metric in METRICS:
            values = np.array([row[metric] for row in trials], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(summaries: list[dict], output_dir: Path) -> None:
    """Per-metric comparisons; retain filenames and add named main frame figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "frame_error": "Raw frame error (Frobenius norm)",
        "normalized_frame_error": r"Normalized frame error ($\mathrm{error}/\sqrt{d}$)",
        "rotation_error": "Registration rotation error (Frobenius norm)",
        "normalized_rotation_error": r"Normalized rotation error ($\mathrm{error}/\sqrt{d}$)",
    }
    dimensions = sorted({row["dimension"] for row in summaries})
    counts = [row["num_trials"] for row in summaries]
    count_label = f"{counts[0]} trials per dataset/dimension" if len(set(counts)) == 1 else "trial counts vary"
    colors = {"gaussian_ellipsoid": "#235789", "uniform_box": "#d67c1c", "laplace_product": "#37805b"}
    markers = {"gaussian_ellipsoid": "o", "uniform_box": "s", "laplace_product": "^"}
    for metric, ylabel in labels.items():
        fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        missing_std = False
        for dataset_type in DATASET_TYPES:
            series = sorted((row for row in summaries if row["dataset_type"] == dataset_type),
                            key=lambda row: row["dimension"])
            if not series:
                continue
            dims = np.array([row["dimension"] for row in series])
            means = np.array([row[f"{metric}_mean"] for row in series])
            stds = np.array([row[f"{metric}_std"] for row in series])
            ax.plot(dims, means, marker=markers[dataset_type], color=colors[dataset_type],
                    linewidth=2, label=DATASET_LABELS[dataset_type])
            finite = np.isfinite(stds)
            if finite.any():
                ax.errorbar(dims[finite], means[finite], yerr=stds[finite],
                            fmt="none", capsize=5, color=colors[dataset_type])
            missing_std |= not finite.all()
        ax.set(xlabel="Dimension", ylabel=ylabel, xticks=dimensions,
               title=f"Source distribution comparison\n{count_label}; mean ± 1 sample SD")
        ax.grid(alpha=0.25)
        ax.legend()
        if missing_std:
            ax.text(0.02, 0.97, "SD undefined for groups with one trial", transform=ax.transAxes,
                    va="top", fontsize=9)
        fig.savefig(output_dir / f"{metric.replace('_', '-')}-vs-dimension.png", dpi=180)
        if metric in ("frame_error", "normalized_frame_error"):
            fig.savefig(output_dir / f"{metric.replace('_', '-')}-by-dataset-vs-dimension.png", dpi=180)
        plt.close(fig)


def run_experiment(config: ExperimentConfig, output_dir: Path) -> list[dict]:
    config.validate()
    validate_frame_error()
    print("Frame metric validation passed (including frame/rotation distinction).", flush=True)
    validation = validate_generators()
    for check in validation:
        print(
            f"{check['dataset_type']} generator passed: "
            f"max standardized mean={check['max_standardized_mean_error']:.4f}, "
            f"max relative variance error={check['max_relative_variance_error']:.4f}, "
            f"mean marginal kurtosis={check['mean_marginal_kurtosis']:.3f}",
            flush=True,
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "generator_validation.csv", validation)
    rows = []
    # Persist each trial so a long run leaves completed observations on disk.
    with (output_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for dataset_type in config.dataset_types:
            for dimension in config.dimensions:
                for trial in range(config.num_trials):
                    row = run_trial(config, dimension, trial, dataset_type)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                    print(
                        f"{dataset_type} d={dimension:2d} trial={trial + 1:2d}/{config.num_trials} "
                        f"frame={row['frame_error']:.5f} rotation={row['rotation_error']:.5f} "
                        f"MMD={row['final_mmd']:+.7f} starts={row['actual_starts']} "
                        f"unconverged={row['unconverged_starts']} time={row['runtime_seconds']:.2f}s "
                        f"greedy_moves={row['greedy_num_accepted_moves']} "
                        f"greedy_pairs={row['greedy_num_pair_evaluations']} "
                        f"greedy_time={row['greedy_runtime_seconds']:.6f}s "
                        f"greedy_stop={row['greedy_stopping_criterion']}",
                        flush=True,
                    )
    summaries = aggregate_results(rows)
    _write_csv(output_dir / "aggregated_results.csv", summaries)
    save_plots(summaries, output_dir)
    print(f"Saved trial/aggregate/validation CSVs and six plots to {output_dir.resolve()}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-types", nargs="+", choices=DATASET_TYPES, default=list(DATASET_TYPES))
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 3, 5, 10, 20])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--max-greedy-moves", type=int, default=100,
                        help="Safety cap on accepted pair flips; normally stop after a full scan without improvement.")
    parser.add_argument("--greedy-tolerance", type=float, default=1e-12,
                        help="Minimum MMD decrease required to accept a pair flip.")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--pca-max-iterations", type=int, default=200)
    parser.add_argument("--min-gradient-norm", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "numerical_results" / "best_improvement_controlled")
    args = parser.parse_args()
    config = ExperimentConfig(**{k: tuple(v) if k in ("dimensions", "dataset_types") else v
                                 for k, v in vars(args).items() if k != "output_dir"})
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    run_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()
