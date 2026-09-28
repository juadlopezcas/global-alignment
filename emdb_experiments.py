"""Independent-resampling registration experiment for the EMD-9515 XYZ clouds.

The two input files are fixed independent samples of the same density map.
Each trial uses X = X_base.copy(), Y = Y_base.copy() @ R_true.T, with no
additional synthetic noise and no pointwise correspondence assumption.
Rows are points: X_aligned = Xc @ R_est.T.

Registration functions are copied unchanged from numerical_experiments.py:
shared centering/normalization, weighted Riemannian PCA, best-improvement
pairwise sign refinement, and exactly one final TrustRegions MMD solve.
Ground truth is used to construct the observation and for evaluation only;
the fitting functions never receive it.

Run the full 20-trial experiment::

    python emdb_experiments.py

Use a separate directory for a two-trial sanity run::

    python emdb_experiments.py --num-trials 2 --output-dir numerical_results/emdb_9515_independent_resampling_sanity

Runtime covers trial setup, fitting, and evaluation; loading and plotting
are excluded. Sample standard deviation is undefined for a single trial.
No resolution, molecular name, or contour level is inferred from XYZ files.
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


SCRIPT_DIR = Path(__file__).resolve().parent
EMDB_ID = "EMD-9515"
DIMENSION = 3


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


def run_trial(
    config: ExperimentConfig, X_base: np.ndarray, Y_base: np.ndarray, trial: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Return a CSV row and normalized plotting coordinates; never pair points."""
    _validate_clouds(X_base, Y_base)
    if X_base.shape[1] != DIMENSION:
        raise ValueError("This experiment requires 3D point clouds.")
    rotation_rng, fit_rng = trial_rngs(config.seed, trial)
    start_time = perf_counter()
    X = X_base.copy()
    Y_unrotated = Y_base.copy()
    R_true = random_rotation(DIMENSION, rotation_rng)
    assert_rotation(R_true)
    Y = Y_unrotated @ R_true.T
    Xc, Yc, scale = center_and_normalize(X, Y)
    result = register_point_clouds(
        Xc, Yc, fit_rng, sigma=config.sigma,
        max_greedy_moves=config.max_greedy_moves, greedy_tolerance=config.greedy_tolerance,
        max_iterations=config.max_iterations,
        pca_max_iterations=config.pca_max_iterations, min_gradient_norm=config.min_gradient_norm,
    )
    assert_rotation(result.R_est)
    # Evaluation boundary: fitting receives only the observations, RNG, and settings.
    # Independent resampling can cause frame differences even at the true rotation.
    F_Y_true_from_X = R_true @ result.U_X
    frame_error, normalized_frame_error = signed_permutation_frame_error(
        F_Y_true_from_X, result.U_Y,
    )
    rotation_error = float(np.linalg.norm(R_true - result.R_est, ord="fro"))
    normalized_rotation_error = rotation_error / np.sqrt(DIMENSION)
    angle_error = rotation_angle_error_degrees(R_true, result.R_est)
    winner = result.optimization
    row = {
        "emdb_id": EMDB_ID,
        "source_file": str(Path(config.source_cloud).resolve()),
        "resampled_file": str(Path(config.resampled_cloud).resolve()),
        "trial": trial, "seed": config.seed, "dimension": DIMENSION,
        "n_source": len(X), "n_target": len(Y),
        "sigma": config.sigma, "requested_starts": 1, "actual_starts": 1,
        "max_greedy_moves": config.max_greedy_moves, "greedy_tolerance": config.greedy_tolerance,
        "max_iterations": config.max_iterations,
        "pca_max_iterations": config.pca_max_iterations, "min_gradient_norm": config.min_gradient_norm,
        "normalization_scale": scale, "frame_error": frame_error,
        "normalized_frame_error": normalized_frame_error, "rotation_error": rotation_error,
        "normalized_rotation_error": normalized_rotation_error,
        "rotation_angle_error_degrees": angle_error,
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
    for metric in METRICS + ("normalization_scale", "winner_gradient_norm",
                             "pca_X_gradient_norm", "pca_Y_gradient_norm"):
        if not np.isfinite(row[metric]):
            raise RuntimeError(f"Trial {trial}: nonfinite {metric}.")
    alignment = {
        "X": Xc, "Y": Yc, "X_aligned": Xc @ result.R_est.T,
        # Use the SAME scale as this trial, without independently renormalizing.
        "Y_reference": (Y_unrotated - Y_unrotated.mean(axis=0)) / scale,
    }
    return row, alignment


METRICS = (
    "frame_error", "normalized_frame_error", "rotation_error", "normalized_rotation_error",
    "rotation_angle_error_degrees", "final_mmd", "runtime_seconds",
    "greedy_runtime_seconds", "greedy_num_accepted_moves", "greedy_num_pair_evaluations",
)


def aggregate_results(rows: list[dict]) -> list[dict]:
    """One summary across trials; sample SD is undefined for a single trial."""
    if not rows:
        raise ValueError("Need at least one trial to aggregate.")
    summary = {key: rows[0][key] for key in (
        "emdb_id", "source_file", "resampled_file", "seed", "dimension", "n_source", "n_target",
    )}
    summary["num_trials"] = len(rows)
    for metric in METRICS:
        values = np.array([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    angles = np.array([row["rotation_angle_error_degrees"] for row in rows], dtype=float)
    summary["rotation_angle_error_degrees_median"] = float(np.median(angles))
    summary["rotation_angle_error_degrees_min"] = float(angles.min())
    summary["rotation_angle_error_degrees_max"] = float(angles.max())
    return [summary]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    rows: list[dict], alignment_trial0: dict[str, np.ndarray], output_dir: Path,
) -> None:
    """Three simple trial plots and one qualitative view; no effect on fitting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = [row["trial"] + 1 for row in rows]
    angles = [row["rotation_angle_error_degrees"] for row in rows]
    runtimes = [row["runtime_seconds"] for row in rows]
    for values, ylabel, filename in (
        (angles, "Angular rotation error (degrees)", "rotation-angle-error-degrees.png"),
        (runtimes, "Runtime (seconds)", "runtime-per-trial.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(trials, values, "o-", color="#2369a1")
        ax.set(xlabel="Trial (1-based)", ylabel=ylabel, xticks=trials,
               title="EMD-9515: independent resampling")
        # Start at zero so roundoff-level differences are not visually magnified.
        ax.set_ylim(0, max(1.0, max(values) * 1.1))
        ax.grid(alpha=0.25)
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(angles, bins=min(10, len(rows)), range=(0, max(1.0, max(angles) * 1.1)),
            color="#2369a1", edgecolor="white")
    ax.set(xlabel="Angular rotation error (degrees)", ylabel="Number of trials",
           title=f"EMD-9515: angular error distribution ({len(rows)} trials)")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.savefig(output_dir / "rotation-error-distribution.png", dpi=180)
    plt.close(fig)

    clouds = alignment_trial0
    panels = (
        ("(a) Before registration", clouds["X"], clouds["Y"], "X", "Rotated Y"),
        ("(b) After registration", clouds["X_aligned"], clouds["Y"], "Aligned X", "Rotated Y"),
        ("(c) Unrotated reference", clouds["X"], clouds["Y_reference"], "X", "Unrotated Y"),
    )
    limit = 1.05 * max(float(np.abs(cloud).max()) for cloud in clouds.values())
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    for position, (title, first, second, first_label, second_label) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, position, projection="3d")
        ax.scatter(*first.T, s=4, alpha=0.55, color="#2369a1", label=first_label)
        ax.scatter(*second.T, s=4, alpha=0.55, color="#db7621", label=second_label)
        ax.set(title=title, xlabel="x", ylabel="y", zlabel="z",
               xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=45)
        ax.legend(loc="upper right", fontsize=8, markerscale=2)
    fig.suptitle("EMD-9515, trial 0: centered coordinates with one shared normalization scale")
    fig.savefig(output_dir / "emdb_alignment_trial0.png", dpi=180)
    plt.close(fig)


def run_experiment(config: ExperimentConfig, output_dir: Path) -> list[dict]:
    config.validate()
    X_base, Y_base = load_xyz(config.source_cloud), load_xyz(config.resampled_cloud)
    print(f"X_base: {Path(config.source_cloud).resolve()} shape={X_base.shape}", flush=True)
    print(f"Y_base: {Path(config.resampled_cloud).resolve()} shape={Y_base.shape}", flush=True)
    print("Fixed independent samples; random SO(3) rotations; no added noise or point correspondence.", flush=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    alignment_trial0 = None
    # Persist completed trials without retaining all trial point clouds.
    with (output_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for trial in range(config.num_trials):
            row, alignment = run_trial(config, X_base, Y_base, trial)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            if trial == 0:
                alignment_trial0 = alignment
            print(
                f"trial={trial + 1:2d}/{config.num_trials} "
                f"angle={row['rotation_angle_error_degrees']:.4f} deg "
                f"frame={row['frame_error']:.5f} rotation={row['rotation_error']:.5f} "
                f"MMD={row['final_mmd']:+.7f} time={row['runtime_seconds']:.2f}s "
                f"greedy_moves={row['greedy_num_accepted_moves']} "
                f"greedy_pairs={row['greedy_num_pair_evaluations']} "
                f"greedy_stop={row['greedy_stopping_criterion']} "
                f"optimizer_stop={row['winner_stopping_criterion']}",
                flush=True,
            )
    summaries = aggregate_results(rows)
    _write_csv(output_dir / "aggregated_results.csv", summaries)
    save_plots(rows, alignment_trial0, output_dir)
    summary = summaries[0]
    print(
        f"Angular error: mean={summary['rotation_angle_error_degrees_mean']:.4f}, "
        f"sample SD={summary['rotation_angle_error_degrees_std']:.4f}, "
        f"median={summary['rotation_angle_error_degrees_median']:.4f}, "
        f"min={summary['rotation_angle_error_degrees_min']:.4f}, "
        f"max={summary['rotation_angle_error_degrees_max']:.4f} degrees.",
        flush=True,
    )
    print(f"Saved trial/aggregate CSVs and four plots to {output_dir.resolve()}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-cloud", type=Path, default=SCRIPT_DIR / "emd_9515_1000.xyz")
    parser.add_argument("--resampled-cloud", type=Path, default=SCRIPT_DIR / "emd_9515_X_1000.xyz")
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--max-greedy-moves", type=int, default=100,
                        help="Safety cap on accepted pair flips; normally stop after a full scan without improvement.")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--pca-max-iterations", type=int, default=200)
    parser.add_argument("--min-gradient-norm", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "numerical_results" / "emdb_9515_independent_resampling")
    args = parser.parse_args()
    config = ExperimentConfig(**{k: v for k, v in vars(args).items() if k != "output_dir"})
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    run_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()
