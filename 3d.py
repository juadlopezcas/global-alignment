import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions
from pymanopt.autodiff.backends import autograd
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

# Keep the random draws fixed so results are repeatable.
np.random.seed(42)


# ============================================================================
# 3D DATA SETUP
# ============================================================================

# n = 200
# z = np.random.randn(n, 3)

z = np.loadtxt('emd_9515_1000.xyz')
n = z.shape[0]

# Anisotropic scaling to give the point cloud distinct principal axes
A = np.array([[4, 0, 0], 
              [0, 2, 0], 
              [0, 0, 1]])
X = z @ A.T + np.array([2.0, -1.0, 3.0])   # Deliberately shifted

# Build a rotated/noisy version.
theta_x, theta_y, theta_z = 45, 30, -60
R_true = Rotation.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=True).as_matrix()

Y = z @ A.T @ R_true.T + np.array([-3.0, 4.0, 5.0]) + 20 * np.random.randn(n, 3)

# Remove translation so we focus only on rotational alignment.
Xc = X - X.mean(0)
Yc = Y - Y.mean(0)

# ================= UNIFORM SCALE NORMALIZATION =================
# Find the maximum distance from the origin for both point clouds
scale_X = np.max(np.linalg.norm(Xc, axis=1))
scale_Y = np.max(np.linalg.norm(Yc, axis=1))
global_scale = max(scale_X, scale_Y) # Take max to encompass both clouds

# Scale point clouds down to fit within a sphere of radius 1
Xc = Xc / global_scale
Yc = Yc / global_scale

print(f"Data normalized down by a factor of {global_scale:.2f}")
# ========================================================

# Build KDTree on the normalized target data
kdtree_Y = KDTree(Yc)

print("="*70)
print("3D POINT CLOUD ALIGNMENT VIA RIEMANNIAN OPTIMIZATION (3-STEP PIPELINE)")
print("="*70)
print("\nData Generation:")
print(f"  Number of points: {n}")
print(f"  Original X center: {X.mean(0).round(2)}")
print(f"  Original Y center: {Y.mean(0).round(2)}")
print(f"\nTrue Rotation Matrix:\n{np.round(R_true, 4)}")

# ============================================================================
# RIEMANNIAN OPTIMIZATION (WITH PCA & NN REFINEMENT)
# ============================================================================

manifold = SpecialOrthogonalGroup(3)

# Autograd-friendly helpers for the MMD^2 objective.
def rbf_kernel_ag(X, Y, sigma=0.2):
    X_sq = anp.sum(X**2, axis=1, keepdims=True)
    Y_sq = anp.sum(Y**2, axis=1, keepdims=True)
    pairwise_sq_dists = X_sq + Y_sq.T - 2 * anp.dot(X, Y.T)
    return anp.exp(-pairwise_sq_dists / (2 * sigma**2))

def mmd_squared_ag(X, Y, sigma=0.2):
    n = X.shape[0]
    m = Y.shape[0]
    Kxx = rbf_kernel_ag(X, X, sigma)
    Kyy = rbf_kernel_ag(Y, Y, sigma)
    Kxy = rbf_kernel_ag(X, Y, sigma)
    
    mmd2 = (anp.sum(Kxx) - anp.trace(Kxx)) / (n * (n - 1))
    mmd2 += (anp.sum(Kyy) - anp.trace(Kyy)) / (m * (m - 1))
    mmd2 -= 2 * anp.sum(Kxy) / (n * m)
    return mmd2

# NumPy versions used just for evaluation/printing.
def rbf_kernel(X, Y, sigma=0.2):
    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-pairwise_sq_dists / (2 * sigma**2))

def mmd_squared(X, Y, sigma=0.2):
    n = X.shape[0]
    m = Y.shape[0]
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    mmd2 = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
    mmd2 += (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1))
    mmd2 -= 2 * np.sum(Kxy) / (n * m)
    return mmd2

@autograd(manifold)
def cost(R):
    X_rotated = anp.dot(R, Xc.T).T
    return mmd_squared_ag(X_rotated, Yc, sigma=0.2)

problem = Problem(manifold=manifold, cost=cost)

# --- STEP 1: PCA INITIALIZATION ---
print("\n" + "="*70)
print("STEP 1: 3D PCA INITIALIZATION")
print("="*70)
u_X, s_X, vh_X = np.linalg.svd(Xc)
u_Y, s_Y, vh_Y = np.linalg.svd(Yc)
v_X = vh_X.T
v_Y = vh_Y.T

# Force them into valid rotation matrices (det = 1)
if np.linalg.det(v_X) < 0: v_X[:, 2] *= -1
if np.linalg.det(v_Y) < 0: v_Y[:, 2] *= -1

R_base = v_Y @ v_X.T

# In 3D, SVD-matched principal axes have 4 valid sign combinations (ensuring det=1)
# Corresponding to: Identity, 180° rotation around X-axis, 180° rotation around Y-axis, 180° rotation around Z-axis
symmetries_3d = [
    np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]),
    np.array([[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]]),
    np.array([[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]]),
    np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]])
]

pca_candidates = [R_base @ sym for sym in symmetries_3d]
print("Generated 4 PCA-aligned 3D candidate starting points.")

# --- STEP 2: OPTIMIZE THE 4 CANDIDATES ---
print("\n" + "="*70)
print("STEP 2: OPTIMIZATION (4 CANDIDATES)")
print("="*70)

optimizer = TrustRegions(max_iterations=100, min_gradient_norm=1e-6, verbosity=0)
optimized_results = []

for i, R_init in enumerate(pca_candidates):
    result = optimizer.run(problem, initial_point=R_init)
    R_opt_candidate = result.point
    
    Xc_aligned_cand = (R_opt_candidate @ Xc.T).T
    cand_mmd = mmd_squared(Xc_aligned_cand, Yc)
    
    optimized_results.append({
        'R': R_opt_candidate,
        'mmd': cand_mmd,
        'aligned_pts': Xc_aligned_cand
    })
    print(f"Candidate {i+1}: MMD² = {cand_mmd:+.6f}")

# --- STEP 3: BREAK SYMMETRY WITH KDTree ---
print("\nSTEP 3: KDTree NEAREST NEIGHBOR SELECTION (BREAKING SYMMETRY)")
best_nn_error = float('inf')
best_idx = -1

for i, res in enumerate(optimized_results):
    # Use KDTree for fast batch nearest neighbor querying
    distances, _ = kdtree_Y.query(res['aligned_pts'])
    nn_error = np.mean(distances)
    
    print(f"Candidate {i+1}: KDTree NN Error = {nn_error:.4f}")
    if nn_error < best_nn_error:
        best_nn_error = nn_error
        best_idx = i

# Select the final winner
R_opt = optimized_results[best_idx]['R']
final_mmd = optimized_results[best_idx]['mmd']
Xc_aligned = optimized_results[best_idx]['aligned_pts']

print(f"\n-> Selected Candidate {best_idx+1} as the True Global Optimum!")


# ============================================================================
# FINAL KDTree POINT-TO-POINT MATCHING
# ============================================================================
# Extract final matching distances and indices for visualization
final_distances, matched_indices = kdtree_Y.query(Xc_aligned)
mean_matching_error = np.mean(final_distances)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Mean KDTree Matching Distance: {mean_matching_error:.4f}")
print(f"Found Rotation Matrix:\n{np.round(R_opt, 4)}")


# ============================================================================
# RESULTS EVALUATION
# ============================================================================

# Calculate the angle error between the two rotation matrices (Geodesic distance on SO(3))
cos_angle = (np.trace(R_opt.T @ R_true) - 1) / 2.0
cos_angle = np.clip(cos_angle, -1.0, 1.0)
angle_error_3d = np.arccos(cos_angle) * 180 / np.pi

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\nFound Rotation Matrix:\n{np.round(R_opt, 4)}")
print(f"\nAccuracy:")
print(f"  3D Angle error: {angle_error_3d:.4f}°")

initial_mmd = mmd_squared(Xc, Yc)

print(f"\nMMD² (unbiased):")
print(f"  Initial: {initial_mmd:+.6f}")
print(f"  Final:   {final_mmd:+.6f}")
print(f"  Improvement: {(initial_mmd-final_mmd):.6f}")
print("="*70)

# ============================================================================
# 3D VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 6))

# Helper to plot 3d frames
def plot_3d_frame(ax, R, origin=[0,0,0], scale=3.0, linestyle='-', alpha=1.0):
    colors = ['r', 'g', 'b'] # x=red, y=green, z=blue
    axes_pts = R @ np.diag([scale, scale, scale])
    for i in range(3):
        ax.quiver(*origin, *axes_pts[:, i], color=colors[i], 
                  arrow_length_ratio=0.1, linestyle=linestyle, alpha=alpha, linewidth=2)

# 1) Original clouds before alignment.
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(Xc[:, 0], Xc[:, 1], Xc[:, 2], alpha=0.3, label='Xc (Source)', s=20, c='tab:blue')
ax1.scatter(Yc[:, 0], Yc[:, 1], Yc[:, 2], alpha=0.3, label='Yc (Target)', s=20, c='tab:orange')
ax1.set_title(f'Centered Datasets (Before)\nMMD² = {initial_mmd:.4f}', fontweight='bold')
ax1.legend()

# # 2) Centered clouds after alignment.
# ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# ax2.scatter(Xc_aligned[:, 0], Xc_aligned[:, 1], Xc_aligned[:, 2], alpha=0.3, label='R*Xc (Aligned)', s=20, c='tab:blue')
# ax2.scatter(Yc[:, 0], Yc[:, 1], Yc[:, 2], alpha=0.3, label='Yc (Target)', s=20, c='tab:orange')
# ax2.set_title(f'After Alignment\nMMD² = {final_mmd:.4f}', fontweight='bold')
# ax2.legend()

# 2) Centered clouds after alignment.
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(Xc_aligned[:, 0], Xc_aligned[:, 1], Xc_aligned[:, 2], alpha=0.5, label='R*Xc (Aligned)', s=20, c='tab:blue')
ax2.scatter(Yc[:, 0], Yc[:, 1], Yc[:, 2], alpha=0.5, label='Yc (Target)', s=20, c='tab:orange')

# ========================================
# Point-to-Point KDTree Lines
# ========================================
for i in range(n):
    p_x = Xc_aligned[i]
    p_y = Yc[matched_indices[i]]
    ax2.plot([p_x[0], p_y[0]], [p_x[1], p_y[1]], [p_x[2], p_y[2]], 'k-', alpha=0.2, linewidth=0.5)

ax2.set_title(f'After Alignment + KDTree Matching\nMean NN Dist = {mean_matching_error:.4f}', fontweight='bold')
ax2.legend()


# 3) Rotation Coordinate Frames Comparison
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
# Identity / Base frame
plot_3d_frame(ax3, np.eye(3), scale=2, alpha=0.3)
# True Rotation (Solid)
plot_3d_frame(ax3, R_true, scale=3, linestyle='-', alpha=0.8)
# Found Rotation (Dashed)
plot_3d_frame(ax3, R_opt, scale=3.5, linestyle='--', alpha=1.0)

ax3.set_xlim([-3, 3])
ax3.set_ylim([-3, 3])
ax3.set_zlim([-3, 3])
ax3.set_title(f'Coordinate Frames (R vs G vs B = X, Y, Z)\nAngle Error = {angle_error_3d:.2f}°', fontweight='bold')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()
plt.show()