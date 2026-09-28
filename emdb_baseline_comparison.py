"""Controlled EMD-9515 comparison: ours, Open3D ICP, rigid CPD, and Go-ICP.

Standalone copy of the registration core and trial RNG/preprocessing from
emdb_experiments.py. Two fixed independent samples; no point correspondences,
added noise, dataset loop, or dimension loop. In every trial, all four methods
receive the same arrays after ONE shared preprocessing step:
Y = Y_base @ R_true.T; X_aligned = Xc @ R_est.T.

ICP starts at identity. CPD uses its standard identity/zero-translation/unit-
scale initialization and estimates scale, rotation, and translation. Only
rotation is scored by the common accuracy metrics. Neither baseline receives
our initialization or ground truth. Go-ICP alone permits explicitly recorded,
guarded SO(3) cleanup of tiny native numerical drift; other methods are unchanged.

Runtime is measured per method, including adapter/setup and fitting, excluding
shared preprocessing, imports, convention checks, evaluation, and plotting.
This timing scope is shared by all methods; the original EMDB script's timer
also includes trial setup and evaluation, so those runtimes are not identical.

Dependencies (install explicitly in your chosen environment):
    python -m pip install open3d pycpd
Validated with open3d 0.19.0 and pycpd 2.0.0. Nothing is installed by this script.
Go-ICP uses py_goicp from aalavandhaann/go-icp_cython; see GOICP_SETUP.md for
the tested build and source commit. Its full runtime includes DT construction.
Default Go-ICP settings are MSE=0.001, DT=300, no trimming, and 1000 data points.
The centered-data translation search cube is [-0.05,0.05]^3 (configurable).
No Go-ICP-specific recentering or rescaling is performed.

Full benchmark:
    python emdb_baseline_comparison.py --num-trials 20 --seed 42
Smoke test:
    python emdb_baseline_comparison.py --num-trials 2 --output-dir numerical_results/emdb_goicp_smoke_two

Matrix-valued CSV fields are JSON arrays. Success thresholds are descriptive
only. Sample standard deviation is undefined for a one-trial group.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import importlib
from importlib import metadata
import json
import os
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


SCRIPT_DIR = Path(__file__).resolve().parent
EMDB_ID = "EMD-9515"
DIMENSION = 3
METHODS = ("ours", "icp", "cpd", "goicp")
METHOD_LABELS = {"ours": "Ours", "icp": "ICP", "cpd": "Rigid CPD", "goicp": "Go-ICP"}
COMMON_METRICS = (
    "rotation_error", "normalized_rotation_error",
    "rotation_angle_error_degrees", "runtime_seconds",
)


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
    source_cloud: Path = SCRIPT_DIR / "emd_9515_1000.xyz"
    resampled_cloud: Path = SCRIPT_DIR / "emd_9515_X_1000.xyz"
    num_trials: int = 20
    seed: int = 42
    sigma: float = 1.0
    max_greedy_moves: int = 100
    greedy_tolerance: float = 1e-12
    max_iterations: int = 100
    pca_max_iterations: int = 200
    min_gradient_norm: float = 1e-6

    def validate(self) -> None:
        if self.num_trials < 1:
            raise ValueError("num_trials must be positive.")
        if self.seed < 0 or self.max_greedy_moves < 0:
            raise ValueError("Seed and max_greedy_moves must be nonnegative.")
        if not np.isfinite(self.greedy_tolerance) or self.greedy_tolerance < 0:
            raise ValueError("greedy_tolerance must be finite and nonnegative.")
        if self.max_iterations < 1 or self.pca_max_iterations < 1:
            raise ValueError("Optimizer iteration limits must be positive.")
        _validate_sigma(self.sigma)
        if not np.isfinite(self.min_gradient_norm) or self.min_gradient_norm <= 0:
            raise ValueError("min_gradient_norm must be finite and positive.")


def load_xyz(path: Path) -> np.ndarray:
    """Load exactly three coordinates per row; skip blanks and # comments."""
    path = Path(path)
    points = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            coordinates = line.partition("#")[0].split()
            if not coordinates:
                continue
            if len(coordinates) != DIMENSION:
                raise ValueError(f"{path}:{line_number}: expected exactly 3 coordinate columns.")
            try:
                point = [float(value) for value in coordinates]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: coordinates must be numbers.") from exc
            if not np.isfinite(point).all():
                raise ValueError(f"{path}:{line_number}: coordinates must be finite.")
            points.append(point)
    if len(points) < 2:
        raise ValueError(f"{path}: need at least two 3D points.")
    return np.asarray(points, dtype=float)


def trial_rngs(seed: int, trial: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Independent rotation/fit streams, reproducible by master seed and trial."""
    rotation_seed, fit_seed = np.random.SeedSequence([seed, trial]).spawn(2)
    return np.random.default_rng(rotation_seed), np.random.default_rng(fit_seed)


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


def rotation_angle_error_degrees(R_true: np.ndarray, R_est: np.ndarray) -> float:
    """SO(3) relative rotation angle, for evaluation only."""
    R_true, R_est = np.asarray(R_true, dtype=float), np.asarray(R_est, dtype=float)
    if R_true.shape != (3, 3) or R_est.shape != (3, 3):
        raise ValueError("Angular rotation error requires two 3 x 3 rotations.")
    assert_rotation(R_true)
    assert_rotation(R_est)
    cos_theta = (np.trace(R_est.T @ R_true) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


@dataclass(frozen=True)
class BaselineConfig(ExperimentConfig):
    icp_max_correspondence_distance: float = 2.0
    goicp_mse_threshold: float = 0.001
    goicp_dt_size: int = 300
    goicp_trim_fraction: float = 0.0
    goicp_num_data_points: int = 1000
    goicp_translation_min: float = -0.05
    goicp_translation_width: float = 0.10

    def validate(self) -> None:
        super().validate()
        if not np.isfinite(self.icp_max_correspondence_distance) or self.icp_max_correspondence_distance <= 0:
            raise ValueError("icp_max_correspondence_distance must be finite and positive.")
        _validate_goicp_options(self.goicp_mse_threshold, self.goicp_dt_size,
                               self.goicp_trim_fraction, self.goicp_num_data_points,
                               self.goicp_translation_min, self.goicp_translation_width)


@dataclass
class MethodResult:
    R_est: np.ndarray
    runtime_seconds: float
    diagnostics: dict
    registration: RegistrationResult | None = None


def _require_dependency(name: str):
    """Lazy, explicit failure: never install packages from inside the experiment."""
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Cannot load required package '{name}': {exc}. "
            "Install in the Python environment running this script with: "
            "python -m pip install open3d pycpd"
        ) from exc


def _check_method_rotation(R: np.ndarray, method: str) -> None:
    try:
        if np.asarray(R).shape != (3, 3):
            raise ValueError("expected a 3 x 3 rotation")
        assert_rotation(R)
    except ValueError as exc:
        raise RuntimeError(f"{method} returned an invalid SO(3) rotation: {exc}") from exc


def run_ours(Xc: np.ndarray, Yc: np.ndarray, rng: np.random.Generator, config: ExperimentConfig) -> MethodResult:
    start_time = perf_counter()
    result = register_point_clouds(
        Xc, Yc, rng, sigma=config.sigma,
        max_greedy_moves=config.max_greedy_moves, greedy_tolerance=config.greedy_tolerance,
        max_iterations=config.max_iterations,
        pca_max_iterations=config.pca_max_iterations, min_gradient_norm=config.min_gradient_norm,
    )
    runtime = perf_counter() - start_time
    return MethodResult(result.R_est, runtime, {
        "initialization": "pca_best_improvement",
        "final_mmd": result.final_mmd,
        "greedy_runtime_seconds": result.greedy.runtime_seconds,
        "greedy_num_accepted_moves": result.greedy.num_accepted_moves,
        "greedy_num_pair_evaluations": result.greedy.num_pair_evaluations,
        "greedy_converged": result.greedy.converged,
        "greedy_stopping_criterion": result.greedy.stopping_criterion,
        "requested_starts": 1, "actual_starts": 1,
        "winner_iterations": result.optimization.iterations,
        "winner_gradient_norm": result.optimization.gradient_norm,
        "winner_stopping_criterion": result.optimization.stopping_criterion,
        "pca_X_gradient_norm": result.pca_X.gradient_norm,
        "pca_Y_gradient_norm": result.pca_Y.gradient_norm,
        "pca_X_stopping_criterion": result.pca_X.stopping_criterion,
        "pca_Y_stopping_criterion": result.pca_Y.stopping_criterion,
    }, registration=result)


def run_icp(
    Xc: np.ndarray, Yc: np.ndarray, *, max_correspondence_distance: float = 2.0,
    initial_transform: np.ndarray | None = None,
) -> MethodResult:
    """Open3D source=Xc, target=Yc; R_est = returned T[:3, :3], no transpose.

    Open3D applies T to column homogeneous points, hence for rows the mapping
    is Xc @ R_est.T + T[:3, 3]. The optional nonidentity initialization exists
    ONLY for the convention test; the benchmark always uses the identity.
    Point-to-point scaling is disabled and convergence criteria use library
    defaults (Open3D 0.19: 30 iterations, relative fitness/RMSE 1e-6).
    API: https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_icp.html
    """
    o3d = _require_dependency("open3d")
    start_time = perf_counter()
    _validate_clouds(Xc, Yc)
    if Xc.shape[1] != 3:
        raise ValueError("ICP wrapper requires 3D clouds.")
    if not np.isfinite(max_correspondence_distance) or max_correspondence_distance <= 0:
        raise ValueError("ICP maximum correspondence distance must be positive and finite.")
    T_init = np.eye(4) if initial_transform is None else np.array(initial_transform, dtype=float, copy=True)
    if T_init.shape != (4, 4) or not np.isfinite(T_init).all():
        raise ValueError("ICP initialization must be a finite 4 x 4 transformation.")
    _check_method_rotation(T_init[:3, :3], "ICP initialization")
    if not np.allclose(T_init[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("Invalid homogeneous last row in ICP initialization.")
    source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    # Open3D 0.19's NumPy adapter requires a writable buffer. Copy coordinates
    # verbatim into library-owned inputs; do not preprocess or alter the data.
    source.points = o3d.utility.Vector3dVector(np.array(Xc, dtype=float, copy=True))
    target.points = o3d.utility.Vector3dVector(np.array(Yc, dtype=float, copy=True))
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
    fitted = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        criteria,
    )
    T = np.array(fitted.transformation, dtype=float, copy=True)
    runtime = perf_counter() - start_time
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise RuntimeError("ICP returned a nonfinite or malformed 4 x 4 transformation.")
    R_est = T[:3, :3].copy()
    _check_method_rotation(R_est, "ICP")
    return MethodResult(R_est, runtime, {
        "initialization": "identity" if initial_transform is None else "convention_test_transform",
        "icp_transformation": json.dumps(T.tolist()),
        "icp_initial_transformation": json.dumps(T_init.tolist()),
        "icp_translation_norm": float(np.linalg.norm(T[:3, 3])),
        "icp_fitness": float(fitted.fitness), "icp_inlier_rmse": float(fitted.inlier_rmse),
        "icp_max_correspondence_distance": max_correspondence_distance,
        "icp_max_iterations": criteria.max_iteration,
        "icp_relative_fitness": criteria.relative_fitness,
        "icp_relative_rmse": criteria.relative_rmse,
    })


def run_rigid_cpd(Xc: np.ndarray, Yc: np.ndarray) -> MethodResult:
    """pycpd target X=Yc, moving source Y=Xc; return R_library.T.

    pycpd transforms row points as s * Y @ R_library + t. Transposing ONLY
    the returned rotation converts to our s * Xc @ R_est.T + t convention.
    Scale and translation are kept and recorded, never forced away. Standard
    package defaults are used (pycpd 2.0: max_iterations=100, tolerance=0.001,
    w=0, initial R=I, s=1, t=0, with automatic initial variance).
    Source: https://github.com/siavashk/pycpd/blob/master/pycpd/rigid_registration.py
    """
    pycpd = _require_dependency("pycpd")
    start_time = perf_counter()
    _validate_clouds(Xc, Yc)
    if Xc.shape[1] != 3:
        raise ValueError("CPD wrapper requires 3D clouds.")
    # pycpd names its fixed cloud X and its moving cloud Y.
    registration = pycpd.RigidRegistration(X=Yc, Y=Xc)
    transformed, (scale, R_library, translation) = registration.register()
    runtime = perf_counter() - start_time
    R_est = np.asarray(R_library, dtype=float).T.copy()
    _check_method_rotation(R_est, "Rigid CPD")
    translation = np.asarray(translation, dtype=float).reshape(3)
    scale = float(scale)
    if not np.isfinite(scale) or not np.isfinite(translation).all() or not np.isfinite(transformed).all():
        raise RuntimeError("Rigid CPD returned nonfinite scale, translation, or transformed points.")
    # Verify conversion against the package's actual transformed points, without
    # estimating correspondences or changing any returned parameter.
    if not np.allclose(transformed, scale * (Xc @ R_est.T) + translation, atol=1e-10, rtol=1e-10):
        raise RuntimeError("Rigid CPD transformed points disagree with the documented row convention.")
    return MethodResult(R_est, runtime, {
        "initialization": "library_default_identity",
        "cpd_scale": scale, "cpd_translation_norm": float(np.linalg.norm(translation)),
        "cpd_translation": json.dumps(translation.tolist()),
        "cpd_iterations": registration.iteration, "cpd_max_iterations": registration.max_iterations,
        "cpd_tolerance": registration.tolerance, "cpd_w": registration.w,
        "cpd_stopping_criterion": "tolerance" if registration.diff <= registration.tolerance else "max_iterations",
    })


def _require_goicp():
    """Load the selected upstream wrapper; never install or compile at runtime."""
    try:
        module = importlib.import_module("py_goicp")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Go-ICP requires py_goicp from https://github.com/aalavandhaann/go-icp_cython "
            "(distribution py-goicp). Build it with Cython/autowrap and a C++ compiler; "
            "see GOICP_SETUP.md for the tested Windows build. The alternative is the "
            "original C++ executable at https://github.com/yangjiaolong/Go-ICP; "
            "this benchmark uses the Python wrapper and does not silently switch backends. "
            f"Import error: {exc}"
        ) from exc
    required = ("GoICP", "POINT3D", "ROTNODE", "TRANSNODE")
    if any(not hasattr(module, name) for name in required):
        raise RuntimeError("Installed py_goicp does not expose the expected upstream API; see GOICP_SETUP.md.")
    return module


def _validate_goicp_options(
    mse_threshold: float, dt_size: int, trim_fraction: float, num_data_points: int,
    translation_min: float, translation_width: float,
) -> None:
    if not np.isfinite(mse_threshold) or mse_threshold <= 0:
        raise ValueError("goicp_mse_threshold must be positive and finite.")
    if not isinstance(dt_size, (int, np.integer)) or dt_size < 2:
        raise ValueError("goicp_dt_size must be an integer >= 2.")
    if not np.isfinite(trim_fraction) or not 0 <= trim_fraction < 1:
        raise ValueError("goicp_trim_fraction must be in [0, 1).")
    if not isinstance(num_data_points, (int, np.integer)) or num_data_points < 0 or num_data_points == 1:
        raise ValueError("goicp_num_data_points must be 0 (all points) or an integer >= 2.")
    if not np.isfinite(translation_min):
        raise ValueError("goicp_translation_min must be finite.")
    if not np.isfinite(translation_width) or translation_width <= 0:
        raise ValueError("goicp_translation_width must be positive and finite.")


GOICP_ROUNDOFF_LIMIT = 5e-6


class GoICPRotationError(RuntimeError):
    """Preserve native rotation diagnostics when numerical cleanup is refused."""

    def __init__(self, reason: str, diagnostics: dict):
        self.diagnostics = diagnostics
        super().__init__(f"Go-ICP returned an invalid SO(3) rotation: {reason}. Raw diagnostics: {json.dumps(diagnostics)}")


def clean_goicp_rotation(returned) -> tuple[np.ndarray, dict]:
    """Record the raw return; project ONLY if strict validation fails and drift is tiny.

    Trial index 3 exhibited Frobenius residual 1.19322e-6, max residual
    1.05966e-6, |det-1|=7.09068e-7 and max |singular_value-1|=5.90456e-7.
    The fixed 5e-6 guard bounds each diagnostic and the correction distance.
    It is independent of R_true, never widened automatically, and does not
    change the shared assert_rotation tolerance. No transpose or inverse.
    """
    native_array = np.asarray(returned)
    raw = np.array(returned, dtype=float, copy=True)
    diagnostics = {
        "goicp_raw_rotation": json.dumps(raw.tolist()),
        "goicp_raw_dtype": str(native_array.dtype),
        "goicp_raw_return_type": type(returned).__name__,
        "goicp_raw_source": "optimalRotation()",
        "goicp_projection_applied": False, "goicp_projection_distance": 0.0,
        "goicp_cleanup_tolerance": GOICP_ROUNDOFF_LIMIT,
    }
    if raw.shape != (3, 3) or not np.isfinite(raw).all():
        raise GoICPRotationError("expected a finite 3 x 3 matrix; projection refused", diagnostics)
    residual = raw.T @ raw - np.eye(3)
    singular_values = np.linalg.svd(raw, compute_uv=False)
    determinant = float(np.linalg.det(raw))
    diagnostics.update(
        goicp_raw_orthogonality_error_fro=float(np.linalg.norm(residual, ord="fro")),
        goicp_raw_orthogonality_error_max=float(np.max(np.abs(residual))),
        goicp_raw_determinant=determinant,
        goicp_raw_singular_values=json.dumps(singular_values.tolist()),
    )
    deviations = (
        diagnostics["goicp_raw_orthogonality_error_fro"], diagnostics["goicp_raw_orthogonality_error_max"],
        abs(determinant - 1.0), float(np.max(np.abs(singular_values - 1.0))),
    )
    if determinant <= 0 or not np.isfinite(deviations).all() or max(deviations) > GOICP_ROUNDOFF_LIMIT:
        raise GoICPRotationError("raw deviation exceeds the 5e-6 numerical-cleanup guard; projection refused", diagnostics)
    try:
        _check_method_rotation(raw, "Go-ICP")
        return raw, diagnostics  # Preserve all previously accepted rotations exactly.
    except RuntimeError:
        U, _, Vt = np.linalg.svd(raw)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(U @ Vt))
        projected = U @ correction @ Vt
        distance = float(np.linalg.norm(projected - raw, ord="fro"))
        if not np.isfinite(distance) or distance > GOICP_ROUNDOFF_LIMIT:
            raise GoICPRotationError("projection distance exceeds the numerical-cleanup guard", diagnostics)
        _check_method_rotation(projected, "Go-ICP projected")
        diagnostics.update(goicp_projection_applied=True, goicp_projection_distance=distance)
        return projected, diagnostics


def run_goicp(
    Xc: np.ndarray, Yc: np.ndarray, *, mse_threshold: float = 0.001,
    dt_size: int = 300, trim_fraction: float = 0.0, num_data_points: int = 1000,
    translation_min: float = -0.05, translation_width: float = 0.10,
) -> MethodResult:
    """Upstream py_goicp: data=Xc (moving), model=Yc (fixed).

    optimalRotation() is used WITHOUT transposing/inverting: upstream applies
    R to column points, so our row mapping is Xc @ R_est.T + translation.
    clean_goicp_rotation records raw diagnostics and guards any tiny SO(3)
    cleanup before common evaluation; this never affects the native search.
    A synthetic convention check (or its matching saved record) is required.

    No centering/rescaling is done here. POINT3D internally stores float32;
    original coordinate values are forwarded and native storage roundoff is
    reported. Default 1000 uses all EMDB points. A smaller explicit count uses
    the first N source rows, deterministically, without consuming any RNG.
    Count 0 means all rows, and counts above the input size use all rows.

    MSE=0.001, DT=300, expansion=2, no trimming, and the rotation cube
    [-3.1416,3.1416]^3 retain the upstream example settings. The configurable
    translation cube defaults to [-0.05,0.05]^3 for the centered-data experiment.
    Runtime includes conversion/setup, DT construction, search, result
    extraction, and object cleanup. Register() returns DT-based SSE, recorded
    unchanged as goicp_final_error; it is neither MMD nor a rotation error.
    """
    module = _require_goicp()
    start_time = perf_counter()
    _validate_clouds(Xc, Yc)
    if Xc.shape[1] != DIMENSION:
        raise ValueError("Go-ICP requires 3D clouds.")
    _validate_goicp_options(mse_threshold, dt_size, trim_fraction, num_data_points,
                           translation_min, translation_width)
    if np.max(np.abs(Xc)) > 1 + 1e-6 or np.max(np.abs(Yc)) > 1 + 1e-6:
        raise ValueError("Go-ICP requires the shared-normalized coordinates in [-1,1]^3; no extra normalization is applied.")
    count = len(Xc) if num_data_points == 0 else min(num_data_points, len(Xc))
    if int(count * (1 - trim_fraction)) < 2:
        raise ValueError("Go-ICP trimming must retain at least two source points.")
    data = Xc[:count]
    # Go-ICP's original POINT3D API stores floats, not doubles. This is the
    # library's required representation, not another normalization step.
    coordinate_roundoff = max(
        float(np.max(np.abs(cloud.astype(np.float32).astype(float) - cloud)))
        for cloud in (data, Yc)
    )
    model_points = [module.POINT3D(float(x), float(y), float(z)) for x, y, z in Yc]
    data_points = [module.POINT3D(float(x), float(y), float(z)) for x, y, z in data]
    solver = module.GoICP()
    solver.MSEThresh = float(mse_threshold)
    solver.trimFraction = float(trim_fraction)
    solver.doTrim = bool(trim_fraction > 0)
    rotation_node, translation_node = module.ROTNODE(), module.TRANSNODE()
    rotation_node.a = rotation_node.b = rotation_node.c = -3.1416
    rotation_node.w = 6.2832
    translation_node.x = translation_node.y = translation_node.z = float(translation_min)
    translation_node.w = float(translation_width)
    solver.setInitNodeRot(rotation_node)
    solver.setInitNodeTrans(translation_node)
    solver.setDTSizeAndFactor(int(dt_size), 2.0)
    solver.loadModelAndData(len(Yc), model_points, count, data_points)
    dt_start = perf_counter()
    solver.BuildDT()
    dt_runtime = perf_counter() - dt_start
    search_start = perf_counter()
    final_error = float(solver.Register())
    search_runtime = perf_counter() - search_start
    # Read all outputs before native object destruction.
    returned_rotation = solver.optimalRotation()
    translation = np.array(solver.optimalTranslation(), dtype=float, copy=True).reshape(3)
    inlier_count = int(solver.inlierNum)
    del solver
    R_est, rotation_diagnostics = clean_goicp_rotation(returned_rotation)
    runtime = perf_counter() - start_time
    if not np.isfinite(translation).all() or not np.isfinite(final_error):
        raise RuntimeError("Go-ICP returned nonfinite translation or final SSE.")
    return MethodResult(R_est, runtime, {
        "initialization": "goicp_search_bounds",
        "goicp_implementation": "py_goicp (aalavandhaann/go-icp_cython)",
        "goicp_translation": json.dumps(translation.tolist()),
        "goicp_translation_norm": float(np.linalg.norm(translation)),
        "goicp_runtime_seconds": runtime,
        "goicp_dt_runtime_seconds": dt_runtime,
        "goicp_search_runtime_seconds": search_runtime,
        "goicp_mse_threshold": mse_threshold, "goicp_dt_size": dt_size,
        "goicp_trim_fraction": trim_fraction, "goicp_num_data_points": count,
        "goicp_requested_num_data_points": num_data_points,
        "goicp_num_inliers": inlier_count, "goicp_dt_expand_factor": 2.0,
        "goicp_final_error": final_error, "goicp_final_error_type": "distance_transform_SSE",
        "goicp_max_coordinate_roundoff": coordinate_roundoff,
        "goicp_translation_min": translation_min, "goicp_translation_width": translation_width,
        **rotation_diagnostics,
    })


def validate_goicp_convention(config: BaselineConfig) -> dict:
    """Independent easy case; failure stops the benchmark before any EMDB fit."""
    rng = np.random.default_rng(20260928)
    P = rng.uniform([-1.0, -0.5, -0.2], [1.0, 0.8, 0.7], size=(80, 3))
    P[:, 1] += 0.25 * P[:, 0]**2
    theta = np.deg2rad(8.0)
    R_test = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                       [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    Q = P @ R_test.T
    Pc, Qc, _ = center_and_normalize(P, Q)
    result = run_goicp(
        Pc, Qc, mse_threshold=config.goicp_mse_threshold, dt_size=config.goicp_dt_size,
        trim_fraction=config.goicp_trim_fraction, num_data_points=config.goicp_num_data_points,
        translation_min=config.goicp_translation_min, translation_width=config.goicp_translation_width,
    )
    metrics = evaluate_rotation(R_test, result.R_est, result.runtime_seconds)
    translation = np.array(json.loads(result.diagnostics["goicp_translation"]))
    if metrics["rotation_error"] >= 1e-3 or metrics["rotation_angle_error_degrees"] >= 0.1:
        raise RuntimeError(f"Go-ICP convention sanity test failed: {metrics}. No benchmark will run.")
    if np.linalg.norm(translation) >= 1e-3 or not np.allclose(
        Pc @ result.R_est.T + translation, Qc, atol=1e-3, rtol=0,
    ):
        raise RuntimeError("Go-ICP convention sanity test failed on transformed points/translation; benchmark stopped.")
    return {"method": "goicp", **metrics, **result.diagnostics}


def evaluate_rotation(R_true: np.ndarray, R_est: np.ndarray, runtime_seconds: float) -> dict:
    """The SAME evaluation function for every method; no signed-permutation matching."""
    _check_method_rotation(R_true, "Ground truth")
    _check_method_rotation(R_est, "Registration")
    rotation_error = float(np.linalg.norm(R_true - R_est, ord="fro"))
    metrics = {
        "rotation_error": rotation_error,
        "normalized_rotation_error": rotation_error / np.sqrt(DIMENSION),
        "rotation_angle_error_degrees": rotation_angle_error_degrees(R_true, R_est),
        "runtime_seconds": runtime_seconds,
    }
    if not all(np.isfinite(value) for value in metrics.values()) or runtime_seconds < 0:
        raise RuntimeError("Common evaluation metrics must be finite, with nonnegative runtime.")
    return metrics


def validate_baseline_conventions() -> list[dict]:
    """Easy synthetic tests, isolated from every trial's RNG and initialization."""
    rng = np.random.default_rng(20260927)
    P = rng.uniform([-1.0, -0.5, -0.2], [1.0, 0.8, 0.7], size=(80, 3))
    P[:, 1] += 0.25 * P[:, 0]**2
    P -= P.mean(axis=0)
    theta = np.deg2rad(8.0)
    R_test = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                       [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    assert_rotation(R_test)
    Q = P @ R_test.T
    T_test = np.eye(4)
    T_test[:3, :3] = R_test
    # Known transform ONLY in this ICP convention test, never in run_trial.
    icp = run_icp(P, Q, initial_transform=T_test)
    cpd = run_rigid_cpd(P, Q)
    checks = []
    for method, result in (("icp", icp), ("cpd", cpd)):
        _check_method_rotation(result.R_est, method)
        np.testing.assert_allclose(result.R_est, R_test, atol=1e-4, rtol=0,
                                   err_msg=f"{method}: incorrect rotation/convention on the easy test.")
        np.testing.assert_allclose(P @ result.R_est.T, Q, atol=1e-4, rtol=0)
        if method == "icp":
            T = np.array(json.loads(result.diagnostics["icp_transformation"]))
            np.testing.assert_allclose(P @ T[:3, :3].T + T[:3, 3], Q, atol=1e-8, rtol=0)
        else:
            np.testing.assert_allclose(result.diagnostics["cpd_scale"], 1.0, atol=1e-4, rtol=0)
            np.testing.assert_allclose(result.diagnostics["cpd_translation_norm"], 0.0, atol=1e-4, rtol=0)
        checks.append({"method": method, **evaluate_rotation(R_test, result.R_est, result.runtime_seconds)})
    return checks


DIAGNOSTIC_FIELDS = (
    "initialization", "frame_error", "normalized_frame_error", "final_mmd",
    "greedy_runtime_seconds", "greedy_num_accepted_moves", "greedy_num_pair_evaluations",
    "greedy_converged", "greedy_stopping_criterion", "requested_starts", "actual_starts",
    "winner_iterations", "winner_gradient_norm", "winner_stopping_criterion",
    "pca_X_gradient_norm", "pca_Y_gradient_norm", "pca_X_stopping_criterion", "pca_Y_stopping_criterion",
    "icp_transformation", "icp_initial_transformation", "icp_translation_norm",
    "icp_fitness", "icp_inlier_rmse", "icp_max_correspondence_distance",
    "icp_max_iterations", "icp_relative_fitness", "icp_relative_rmse",
    "cpd_scale", "cpd_translation_norm", "cpd_translation", "cpd_iterations",
    "cpd_max_iterations", "cpd_tolerance", "cpd_w", "cpd_stopping_criterion",
)
GOICP_DIAGNOSTIC_FIELDS = (
    "goicp_implementation", "goicp_translation", "goicp_translation_norm", "goicp_runtime_seconds",
    "goicp_dt_runtime_seconds", "goicp_search_runtime_seconds", "goicp_mse_threshold",
    "goicp_dt_size", "goicp_trim_fraction", "goicp_num_data_points", "goicp_requested_num_data_points",
    "goicp_num_inliers", "goicp_dt_expand_factor", "goicp_final_error", "goicp_final_error_type",
    "goicp_max_coordinate_roundoff",
    "goicp_translation_min", "goicp_translation_width",
    "goicp_raw_rotation", "goicp_raw_dtype", "goicp_raw_return_type", "goicp_raw_source",
    "goicp_raw_orthogonality_error_fro", "goicp_raw_orthogonality_error_max",
    "goicp_raw_determinant", "goicp_raw_singular_values", "goicp_projection_applied",
    "goicp_projection_distance", "goicp_cleanup_tolerance",
)
DIAGNOSTIC_FIELDS += GOICP_DIAGNOSTIC_FIELDS


def run_trial(config: BaselineConfig, X_base: np.ndarray, Y_base: np.ndarray, trial: int) -> list[dict]:
    """Generate ONE trial; all methods receive identical preprocessed arrays."""
    _validate_clouds(X_base, Y_base)
    if X_base.shape[1] != DIMENSION:
        raise ValueError("This experiment requires 3D point clouds.")
    # Identical draw order and separate rotation/fit streams as emdb_experiments.py.
    rotation_rng, fit_rng = trial_rngs(config.seed, trial)
    X = X_base.copy()
    Y_unrotated = Y_base.copy()
    R_true = random_rotation(DIMENSION, rotation_rng)
    assert_rotation(R_true)
    Y = Y_unrotated @ R_true.T
    Xc, Yc, scale = center_and_normalize(X, Y)
    # These arrays are shared, never re-generated or independently normalized.
    # Read-only flags detect accidental in-place modifications by any method.
    Xc.setflags(write=False)
    Yc.setflags(write=False)
    runners = (
        ("ours", lambda: run_ours(Xc, Yc, fit_rng, config)),
        ("icp", lambda: run_icp(Xc, Yc, max_correspondence_distance=config.icp_max_correspondence_distance)),
        ("cpd", lambda: run_rigid_cpd(Xc, Yc)),
        ("goicp", lambda: run_goicp(
            Xc, Yc, mse_threshold=config.goicp_mse_threshold, dt_size=config.goicp_dt_size,
            trim_fraction=config.goicp_trim_fraction, num_data_points=config.goicp_num_data_points,
            translation_min=config.goicp_translation_min, translation_width=config.goicp_translation_width,
        )),
    )
    rows = []
    for method, fit in runners:
        try:
            result = fit()  # Ground truth is never an input to these calls.
        except GoICPRotationError as exc:
            exc.diagnostics["trial"] = trial
            raise
        row = {
            "emdb_id": EMDB_ID, "trial": trial, "seed": config.seed, "method": method,
            "dimension": DIMENSION, "n_source": len(Xc), "n_target": len(Yc),
            "source_file": str(Path(config.source_cloud).resolve()),
            "resampled_file": str(Path(config.resampled_cloud).resolve()),
            "normalization_scale": scale,
            # Storing rotations makes shared ground truth and convention auditable.
            "R_true": json.dumps(R_true.tolist()), "R_est": json.dumps(result.R_est.tolist()),
            **evaluate_rotation(R_true, result.R_est, result.runtime_seconds),
            "sigma": config.sigma, "max_greedy_moves": config.max_greedy_moves,
            "greedy_tolerance": config.greedy_tolerance, "max_iterations": config.max_iterations,
            "pca_max_iterations": config.pca_max_iterations, "min_gradient_norm": config.min_gradient_norm,
            **{field: None for field in DIAGNOSTIC_FIELDS},
        }
        row.update(result.diagnostics)
        if method == "ours":
            # Same optional frame diagnostics as the reference EMDB script.
            frame_error, normalized_frame_error = signed_permutation_frame_error(
                R_true @ result.registration.U_X, result.registration.U_Y,
            )
            row.update(frame_error=frame_error, normalized_frame_error=normalized_frame_error)
        for field in DIAGNOSTIC_FIELDS:
            value = row[field]
            if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                raise RuntimeError(f"Trial {trial}, {method}: nonfinite diagnostic {field}.")
        rows.append(row)
    return rows


def aggregate_results(rows: list[dict]) -> list[dict]:
    """Per-method descriptive statistics; thresholds are not literature standards."""
    summaries = []
    for method in METHODS:
        trials = [row for row in rows if row["method"] == method]
        if not trials:
            raise ValueError(f"No results for method {method}.")
        summary = {"emdb_id": EMDB_ID, "method": method, "num_trials": len(trials)}
        for metric in COMMON_METRICS:
            values = np.array([row[metric] for row in trials], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite common metric {metric} for {method}.")
            summary.update({
                f"{metric}_mean": float(values.mean()),
                f"{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                f"{metric}_median": float(np.median(values)),
                f"{metric}_min": float(values.min()), f"{metric}_max": float(values.max()),
            })
        angles = np.array([row["rotation_angle_error_degrees"] for row in trials])
        for threshold in (1, 5, 10):
            summary[f"success_rate_angle_lt_{threshold}_deg"] = float(np.mean(angles < threshold))
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(rows: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ours": "#2369a1", "icp": "#db7621", "cpd": "#26845b", "goicp": "#8751a1"}
    for metric, ylabel, filename in (
        ("rotation_angle_error_degrees", "Angular rotation error (degrees)", "baseline-angle-error-boxplot.png"),
        ("runtime_seconds", "Runtime (seconds)", "baseline-runtime-boxplot.png"),
    ):
        values = [[row[metric] for row in rows if row["method"] == method] for method in METHODS]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        boxes = ax.boxplot(values, patch_artist=True, medianprops={"color": "black"})
        ax.set_xticks(range(1, len(METHODS) + 1), [METHOD_LABELS[method] for method in METHODS])
        for box, method in zip(boxes["boxes"], METHODS):
            box.set_facecolor(colors[method])
            box.set_alpha(0.65)
        ax.set(ylabel=ylabel, title="EMD-9515: independent-resampling comparison")
        ax.set_ylim(0, max(1.0, max(max(group) for group in values) * 1.1))
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for method in METHODS:
        trials = sorted((row for row in rows if row["method"] == method), key=lambda row: row["trial"])
        ax.plot([row["trial"] + 1 for row in trials],
                [row["rotation_angle_error_degrees"] for row in trials],
                "o-", color=colors[method], label=METHOD_LABELS[method])
    ax.set(xlabel="Trial (1-based)", ylabel="Angular rotation error (degrees)",
           xticks=sorted({row["trial"] + 1 for row in rows}),
           title="EMD-9515: identical inputs within each trial")
    ax.set_ylim(0, max(1.0, max(row["rotation_angle_error_degrees"] for row in rows) * 1.1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "baseline-angle-error-by-trial.png", dpi=180)
    plt.close(fig)


def load_goicp_convention_record(path: Path, config: BaselineConfig) -> dict:
    """Reuse a successful saved synthetic check only for identical settings."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("status") != "completed" or record.get("worker_exitcode") != 0:
        raise ValueError("Go-ICP convention record must describe a completed, successful check.")
    check = record.get("result", {})
    if check.get("method") != "goicp":
        raise ValueError("Expected a saved Go-ICP synthetic convention check.")
    expected = {
        "goicp_mse_threshold": config.goicp_mse_threshold,
        "goicp_dt_size": config.goicp_dt_size,
        "goicp_trim_fraction": config.goicp_trim_fraction,
        "goicp_requested_num_data_points": config.goicp_num_data_points,
        "goicp_translation_min": config.goicp_translation_min,
        "goicp_translation_width": config.goicp_translation_width,
        "goicp_dt_expand_factor": 2.0,
    }
    for key, value in expected.items():
        if check.get(key) != value:
            raise ValueError(f"Go-ICP convention record mismatch for {key}: {check.get(key)} != {value}.")
    for key, limit in (("rotation_error", 1e-3), ("rotation_angle_error_degrees", 0.1),
                       ("goicp_translation_norm", 1e-3)):
        value = check.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value < limit:
            raise ValueError(f"Go-ICP convention record did not pass {key}.")
    return check


def load_resume_rows(path: Path, config: BaselineConfig, X_base: np.ndarray, Y_base: np.ndarray) -> list[dict]:
    """Validate a complete trial prefix without refitting or editing its source CSV.

    Existing metrics are retained. Legacy Go-ICP R_est values were raw native
    returns; backfill their raw diagnostics only if the new policy leaves the
    saved rotation unchanged. A conflicting cleanup policy is never hidden.
    """
    path = Path(path)
    saved = json.loads((path.parent / "baseline_run_metadata.json").read_text(encoding="utf-8-sig"))
    for key, value in asdict(config).items():
        if key == "num_trials":
            continue
        previous = saved["config"].get(key)
        matches = Path(previous).resolve() == Path(value).resolve() if key in ("source_cloud", "resampled_cloud") else previous == value
        if not matches:
            raise ValueError(f"Resume configuration mismatch for {key}.")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) % len(METHODS):
        raise ValueError("Resume CSV must contain a nonempty prefix of complete four-method trials.")
    completed = len(rows) // len(METHODS)
    if completed > config.num_trials:
        raise ValueError("Resume CSV already contains more trials than requested.")
    expected = [(trial, method) for trial in range(completed) for method in METHODS]
    if [(int(row["trial"]), row["method"]) for row in rows] != expected:
        raise ValueError("Resume CSV has missing, duplicate, unordered, or non-prefix trial rows.")
    for trial in range(completed):
        rotation_rng, _ = trial_rngs(config.seed, trial)
        R_true = random_rotation(DIMENSION, rotation_rng)
        _, _, scale = center_and_normalize(X_base, Y_base @ R_true.T)
        for row in rows[trial * len(METHODS):(trial + 1) * len(METHODS)]:
            row["trial"] = trial
            if (int(row["seed"]) != config.seed or int(row["n_source"]) != len(X_base)
                    or int(row["n_target"]) != len(Y_base)
                    or not np.isclose(float(row["normalization_scale"]), scale, atol=1e-12, rtol=0)
                    or not np.array_equal(np.array(json.loads(row["R_true"])), R_true)):
                raise ValueError(f"Resume trial {trial} does not match the recorded input/RNG protocol.")
            for key in COMMON_METRICS:
                row[key] = float(row[key])
            R_est = np.array(json.loads(row["R_est"]))
            metrics = evaluate_rotation(R_true, R_est, row["runtime_seconds"])
            if any(not np.isclose(row[key], metrics[key], atol=1e-10, rtol=0) for key in COMMON_METRICS):
                raise ValueError(f"Resume trial {trial} has inconsistent saved rotation metrics.")
            for key in DIAGNOSTIC_FIELDS:
                row.setdefault(key, None)
            if row["method"] == "goicp":
                raw_record = row.get("goicp_raw_rotation")
                raw = json.loads(raw_record or row["R_est"])
                cleaned, diagnostics = clean_goicp_rotation(raw)
                if not np.allclose(cleaned, R_est, atol=1e-12, rtol=0):
                    raise ValueError(f"Resume trial {trial} would require changing its saved rotation; review it explicitly.")
                if not raw_record:
                    diagnostics["goicp_raw_source"] = "legacy CSV R_est (pre-cleanup native return)"
                    diagnostics["goicp_raw_return_type"] = "not captured in legacy CSV"
                    row.update(diagnostics)
    return rows


def run_experiment(
    config: BaselineConfig, output_dir: Path, *, goicp_convention_record: Path | None = None,
    resume_from: Path | None = None,
) -> list[dict]:
    config.validate()
    # Import/preflight before timing so first-call import costs are excluded.
    for package in ("open3d", "pycpd"):
        _require_dependency(package)
    goicp_module = _require_goicp()
    X_base, Y_base = load_xyz(config.source_cloud), load_xyz(config.resampled_cloud)
    output_dir = Path(output_dir)
    if resume_from is not None and output_dir.exists():
        raise ValueError("Resuming requires a NEW output directory; the original results are never overwritten.")
    if (output_dir / "baseline_trial_results.csv").exists() and (output_dir / "baseline_trial_results.csv").stat().st_size:
        raise ValueError("Output already contains trial results; choose a new directory or resume into a new directory.")
    rows = load_resume_rows(resume_from, config, X_base, Y_base) if resume_from is not None else []
    first_trial = len(rows) // len(METHODS)
    print(f"X_base shape={X_base.shape}; Y_base shape={Y_base.shape}", flush=True)
    checks = validate_baseline_conventions()
    if goicp_convention_record is None:
        print(f"Checking Go-ICP convention (DT={config.goicp_dt_size}, MSE={config.goicp_mse_threshold})...", flush=True)
        checks.append(validate_goicp_convention(config))
    else:
        checks.append(load_goicp_convention_record(goicp_convention_record, config))
        print(f"Reusing validated Go-ICP convention: {Path(goicp_convention_record).resolve()}", flush=True)
    for check in checks:
        print(f"{check['method']} convention passed: angle={check['rotation_angle_error_degrees']:.8f} deg", flush=True)
    versions = {name: metadata.version(name) for name in
                ("numpy", "scipy", "pymanopt", "autograd", "matplotlib", "open3d", "pycpd")}
    try:
        versions["py-goicp"] = metadata.version("py-goicp")
    except metadata.PackageNotFoundError:
        versions["py-goicp"] = "unregistered local build"
    print(f"Dependencies: Open3D {versions['open3d']}; pycpd {versions['pycpd']}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "config": asdict(config), "package_versions": versions, "convention_checks": checks,
        "goicp_convention_record": str(Path(goicp_convention_record).resolve()) if goicp_convention_record else None,
        "resume_from": str(Path(resume_from).resolve()) if resume_from else None,
        "resumed_trial_count": first_trial,
        "goicp_rotation_cleanup": {
            "policy": "preserve strict-valid raw rotations; guarded SVD projection only on strict validation failure",
            "guard_tolerance": GOICP_ROUNDOFF_LIMIT,
            "guard_quantities": "Frobenius/max orthogonality residuals, abs(det-1), max abs(singular_values-1), projection distance",
        },
        "runtime_scope": "per-method adapter/setup and fitting; excludes shared preprocessing, imports, evaluation and plotting",
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "success_thresholds": "descriptive only; strict angular error < 1, < 5, < 10 degrees",
        "protocol": "fixed independent samples; one random rotation and one shared preprocessing per trial",
        "goicp_implementation": {
            "backend": "py_goicp", "module_path": str(goicp_module.__file__),
            "repository": "https://github.com/aalavandhaann/go-icp_cython",
            "runtime_scope": "conversion/setup + DT construction + search + result extraction + cleanup",
            "final_error_definition": "native distance-transform SSE returned by Register()",
            "coordinate_storage": "upstream POINT3D uses float32; no recentering or rescaling",
            "subset_rule": "first N source rows in original order; 0 means all; counts above input size use all",
        },
    }
    with (output_dir / "baseline_run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2, default=str, allow_nan=False)
    with (output_dir / "baseline_trial_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            print(f"Preserved {len(rows)} existing rows; resuming at zero-based trial {first_trial}.", flush=True)
        for trial in range(first_trial, config.num_trials):
            for row in run_trial(config, X_base, Y_base, trial):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                print(
                    f"trial={trial + 1}/{config.num_trials} method={row['method']:4s} "
                    f"angle={row['rotation_angle_error_degrees']:.4f} deg "
                    f"rotation={row['rotation_error']:.5f} runtime={row['runtime_seconds']:.3f}s",
                    flush=True,
                )
    summaries = aggregate_results(rows)
    _write_csv(output_dir / "baseline_aggregated_results.csv", summaries)
    save_plots(rows, output_dir)
    print(f"Saved comparison CSVs, metadata, and three plots to {output_dir.resolve()}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-cloud", type=Path, default=SCRIPT_DIR / "emd_9515_1000.xyz")
    parser.add_argument("--resampled-cloud", type=Path, default=SCRIPT_DIR / "emd_9515_X_1000.xyz")
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--max-greedy-moves", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--pca-max-iterations", type=int, default=200)
    parser.add_argument("--min-gradient-norm", type=float, default=1e-6)
    parser.add_argument("--icp-max-correspondence-distance", type=float, default=2.0)
    parser.add_argument("--goicp-mse-threshold", type=float, default=0.001)
    parser.add_argument("--goicp-dt-size", type=int, default=300,
                        help="DT nodes per axis; 300 is the upstream example default and is never lowered automatically.")
    parser.add_argument("--goicp-trim-fraction", type=float, default=0.0)
    parser.add_argument("--goicp-num-data-points", type=int, default=1000,
                        help="Maximum source rows for Go-ICP, taken in file order; 0 uses all. Default uses all 1000 EMDB points.")
    parser.add_argument("--goicp-translation-min", type=float, default=-0.05,
                        help="Translation search cube minimum on each axis for centered clouds.")
    parser.add_argument("--goicp-translation-width", type=float, default=0.10,
                        help="Translation search cube width on each axis.")
    parser.add_argument("--goicp-convention-record", type=Path,
                        help="Reuse a successful saved synthetic convention status JSON with identical Go-ICP settings.")
    parser.add_argument("--resume-from", type=Path,
                        help="Copy complete existing trial rows and run only missing trials into a NEW output directory.")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "numerical_results" / "emdb_baseline_comparison")
    args = parser.parse_args()
    config = BaselineConfig(**{k: v for k, v in vars(args).items()
                              if k not in ("output_dir", "goicp_convention_record", "resume_from")})
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    try:
        run_experiment(config, args.output_dir, goicp_convention_record=args.goicp_convention_record,
                       resume_from=args.resume_from)
    except GoICPRotationError as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / f"goicp_rotation_failure_trial_{exc.diagnostics.get('trial', 'unknown')}.json"
        failure_path.write_text(json.dumps({"error": str(exc), "diagnostics": exc.diagnostics}, indent=2), encoding="utf-8")
        parser.exit(1, f"error: {exc}\nRaw diagnostics saved to {failure_path}\n")
    except (RuntimeError, OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
