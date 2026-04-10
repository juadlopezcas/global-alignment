import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
from pymanopt.manifolds import SpecialOrthogonalGroup, Stiefel
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions
from pymanopt.autodiff.backends import autograd
from scipy.spatial.distance import cdist

from scipy.optimize import linear_sum_assignment

# Keep the random draws fixed so results are repeatable.
np.random.seed(42)

# Main experiment setting.
n = 200  # number of samples

# ============================================================================
# DATA SETUP
# ============================================================================

# Build the first point cloud.
z = np.random.randn(n, 2)
A = np.array([[3, 0], [0, 1]])
X = z @ A.T + np.array([2.0, -1.0])   # Deliberately shifted away from the origin.

# Build a rotated/noisy version of the same base shape.
theta = 45 * np.pi/180
R_true = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
Y = z @ A.T @ R_true.T + np.array([-3.0, 4.0]) + 0.05*np.random.randn(n, 2)

# Remove translation so we focus only on rotational alignment.
Xc = X - X.mean(0)
Yc = Y - Y.mean(0)

print("="*70)
print("POINT CLOUD ALIGNMENT VIA RIEMANNIAN OPTIMIZATION")
print("="*70)
print("\nData Generation:")
print(f"  Number of points: {n}")
print(f"  Original X center: {X.mean(0)}")
print(f"  Original Y center: {Y.mean(0)}")
print(f"\nTrue Rotation:")
print(f"  Angle: {theta * 180/np.pi:.2f}°")
print(f"  Matrix:\n{R_true}")

# ============================================================================
# RIEMANNIAN OPTIMIZATION
# ============================================================================

# O(2) can be represented as Stiefel(2, 2): all 2x2 orthogonal matrices.
# manifold = Stiefel(2, 2)
manifold = SpecialOrthogonalGroup(2)
# Autograd-friendly helpers for the MMD^2 objective.
def rbf_kernel_ag(X, Y, sigma=1.0):
    """RBF (Gaussian) kernel using autograd numpy"""
    # Pairwise squared distances for the Gaussian kernel.
    X_sq = anp.sum(X**2, axis=1, keepdims=True)
    Y_sq = anp.sum(Y**2, axis=1, keepdims=True)
    pairwise_sq_dists = X_sq + Y_sq.T - 2 * anp.dot(X, Y.T)
    return anp.exp(-pairwise_sq_dists / (2 * sigma**2))

def mmd_squared_ag(X, Y, sigma=1.0):
    """Maximum Mean Discrepancy squared using autograd numpy (unbiased)"""
    n = X.shape[0]
    m = Y.shape[0]
    
    # Within-X similarities.
    Kxx = rbf_kernel_ag(X, X, sigma)
    # Within-Y similarities.
    Kyy = rbf_kernel_ag(Y, Y, sigma)
    # Cross-similarities.
    Kxy = rbf_kernel_ag(X, Y, sigma)
    
    # Unbiased MMD^2 estimate (can dip below zero with finite samples).
    mmd2 = (anp.sum(Kxx) - anp.trace(Kxx)) / (n * (n - 1))
    mmd2 += (anp.sum(Kyy) - anp.trace(Kyy)) / (m * (m - 1))
    mmd2 -= 2 * anp.sum(Kxy) / (n * m)
    
    return mmd2

# Objective minimized on the manifold.
@autograd(manifold)
def cost(R):
    """Cost function: MMD²(R @ Xc.T, Yc.T)"""
    X_rotated = anp.dot(R, Xc.T).T
    return mmd_squared_ag(X_rotated, Yc, sigma=1.0)

# Bundle manifold + objective into a pymanopt problem.
problem = Problem(manifold=manifold, cost=cost)

# Solve with trust regions on the Stiefel manifold.
print("\n" + "="*70)
print("OPTIMIZATION")
print("="*70)
print("\nRunning Trust Regions optimizer on Stiefel manifold...")
optimizer = TrustRegions(max_iterations=100, min_gradient_norm=1e-6)
result = optimizer.run(problem)
R_opt = result.point

# Convert the learned rotation matrix into an angle for reporting.
angle_opt = np.arctan2(R_opt[1, 0], R_opt[0, 0])
angle_error = np.abs(angle_opt - theta) * 180/np.pi

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print("\nFound Rotation:")
print(f"  Angle: {angle_opt * 180/np.pi:.2f}°")
print(f"  Matrix:\n{R_opt}")
print(f"\nAccuracy:")
print(f"  Angle error: {angle_error:.4f}°")

# Rotate Xc using the optimizer output.
Xc_aligned = (R_opt @ Xc.T).T

# NumPy versions used just for final evaluation/printing.
def rbf_kernel(X, Y, sigma=1.0):
    """RBF (Gaussian) kernel"""
    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-pairwise_sq_dists / (2 * sigma**2))

def mmd_squared(X, Y, sigma=1.0):
    """Maximum Mean Discrepancy squared (unbiased)"""
    n = X.shape[0]
    m = Y.shape[0]
    
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    
    mmd2 = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
    mmd2 += (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1))
    mmd2 -= 2 * np.sum(Kxy) / (n * m)
    
    return mmd2

initial_mmd = mmd_squared(Xc, Yc)
final_mmd = mmd_squared(Xc_aligned, Yc)

print(f"\nMMD² (unbiased):")
print(f"  Initial: {initial_mmd:+.6f}")
print(f"  Final:   {final_mmd:+.6f}")
print(f"  Improvement: {(initial_mmd-final_mmd):.6f}")
print("="*70)

# ============================================================================
# HUNGARIAN ALGORITHM: POINT-TO-POINT MATCHING
# ============================================================================
print("\n" + "="*70)
print("POINT-TO-POINT MATCHING (HUNGARIAN ALGORITHM)")
print("="*70)

# Calculate the distance matrix (Cost Matrix)
cost_matrix = cdist(Xc_aligned, Yc, metric='euclidean')

# Solve for the global optimal one-to-one matching using the Hungarian algorithm
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Extract matching errors
matched_distances = cost_matrix[row_ind, col_ind]
mean_matching_error = np.mean(matched_distances)

print(f"Successfully found {len(row_ind)} one-to-one matching pairs.")
print(f"Mean point-to-point matching distance (error): {mean_matching_error:.4f}")

# 4. Visualize
plt.scatter(Xc_aligned[:, 0], Xc_aligned[:, 1], alpha=0.6, label='R*Xc (aligned)', s=30, c='tab:blue')
plt.scatter(Yc[:, 0], Yc[:, 1], alpha=0.6, label='Yc (target)', s=30, c='tab:orange')

# Iterate through all matching pairs and connect them with thin lines
# Since there are many points, alpha (transparency) is set to 0.2 to avoid visual clutter
for i in range(len(row_ind)):
    p_x = Xc_aligned[row_ind[i]]
    p_y = Yc[col_ind[i]]
    plt.plot([p_x[0], p_y[0]], [p_x[1], p_y[1]], 'k-', alpha=0.2, linewidth=0.8)

plt.title(f'Hungarian Algorithm: Explicit Point Matches\nMean Matching Distance = {mean_matching_error:.4f}', fontsize=14, fontweight='bold')
plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# 1) Original, not-centered point clouds.
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(X[:, 0], X[:, 1], alpha=0.5, label='X', s=30, c='tab:blue')
ax1.scatter(Y[:, 0], Y[:, 1], alpha=0.5, label='Y', s=30, c='tab:orange')
ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)
ax1.set_title('Original Non-Centered Datasets', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 2) Centered clouds before alignment.
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(Xc[:, 0], Xc[:, 1], alpha=0.5, label='Xc', s=30, c='tab:blue')
ax2.scatter(Yc[:, 0], Yc[:, 1], alpha=0.5, label='Yc', s=30, c='tab:orange')
ax2.set_xlabel('x₁', fontsize=12)
ax2.set_ylabel('x₂', fontsize=12)
ax2.set_title(f'Centered Datasets (Before Alignment)\nMMD² = {initial_mmd:.4f}', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# 3) Centered clouds after alignment.
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(Xc_aligned[:, 0], Xc_aligned[:, 1], alpha=0.5, label='R*Xc (aligned)', s=30, c='tab:blue')
ax3.scatter(Yc[:, 0], Yc[:, 1], alpha=0.5, label='Yc (target)', s=30, c='tab:orange')
ax3.set_xlabel('x₁', fontsize=12)
ax3.set_ylabel('x₂', fontsize=12)
ax3.set_title(f'After Alignment\nMMD² = {final_mmd:.4f}', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

# 4) Coordinate frame view of the true rotation.
ax4 = plt.subplot(2, 3, 4)
scale = 2.5
# Draw the original frame.
ax4.arrow(0, 0, scale, 0, head_width=0.2, head_length=0.25, fc='tab:blue', ec='tab:blue', linewidth=3, label='Original frame')
ax4.arrow(0, 0, 0, scale, head_width=0.2, head_length=0.25, fc='tab:blue', ec='tab:blue', linewidth=3)
# Draw the frame after applying the true rotation.
v1_rot = R_true @ np.array([scale, 0])
v2_rot = R_true @ np.array([0, scale])
ax4.arrow(0, 0, v1_rot[0], v1_rot[1], head_width=0.2, head_length=0.25, fc='tab:red', ec='tab:red', linewidth=3, linestyle='--', label='True rotation')
ax4.arrow(0, 0, v2_rot[0], v2_rot[1], head_width=0.2, head_length=0.25, fc='tab:red', ec='tab:red', linewidth=3, linestyle='--')
ax4.set_xlim(-3, 3)
ax4.set_ylim(-3, 3)
ax4.set_xlabel('x₁', fontsize=12)
ax4.set_ylabel('x₂', fontsize=12)
ax4.set_title(f'True Rotation Frame\nAngle = {theta*180/np.pi:.1f}°', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11, loc='upper left')
ax4.grid(True, alpha=0.3)
ax4.axis('equal')
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)

# 5) Coordinate frame view of the recovered rotation.
ax5 = plt.subplot(2, 3, 5)
# Draw the original frame.
ax5.arrow(0, 0, scale, 0, head_width=0.2, head_length=0.25, 
          fc='tab:blue', ec='tab:blue', linewidth=3, label='Original frame')
ax5.arrow(0, 0, 0, scale, head_width=0.2, head_length=0.25, 
          fc='tab:blue', ec='tab:blue', linewidth=3)
# Draw the frame after applying the recovered rotation.
v1_opt = R_opt @ np.array([scale, 0])
v2_opt = R_opt @ np.array([0, scale])
ax5.arrow(0, 0, v1_opt[0], v1_opt[1], head_width=0.2, head_length=0.25, fc='tab:green', ec='tab:green', linewidth=3, linestyle='--', label='Found rotation')
ax5.arrow(0, 0, v2_opt[0], v2_opt[1], head_width=0.2, head_length=0.25, fc='tab:green', ec='tab:green', linewidth=3, linestyle='--')
ax5.set_xlim(-3, 3)
ax5.set_ylim(-3, 3)
ax5.set_xlabel('x₁', fontsize=12)
ax5.set_ylabel('x₂', fontsize=12)
ax5.set_title(f'Found Rotation Frame\nAngle = {angle_opt*180/np.pi:.1f}°', fontsize=14, fontweight='bold')
ax5.legend(fontsize=11, loc='upper left')
ax5.grid(True, alpha=0.3)
ax5.axis('equal')
ax5.axhline(y=0, color='k', linewidth=0.5)
ax5.axvline(x=0, color='k', linewidth=0.5)

# 6) Overlay all frames for an easy visual comparison.
ax6 = plt.subplot(2, 3, 6)
# Original frame.
ax6.arrow(0, 0, scale, 0, head_width=0.2, head_length=0.25, fc='tab:blue', ec='tab:blue', linewidth=3, alpha=0.6, label='Original')
ax6.arrow(0, 0, 0, scale, head_width=0.2, head_length=0.25, fc='tab:blue', ec='tab:blue', linewidth=3, alpha=0.6)
# True rotation frame.
ax6.arrow(0, 0, v1_rot[0], v1_rot[1], head_width=0.2, head_length=0.25, fc='tab:red', ec='tab:red', linewidth=3, linestyle='--', label='True', alpha=0.8)
ax6.arrow(0, 0, v2_rot[0], v2_rot[1], head_width=0.2, head_length=0.25, fc='tab:red', ec='tab:red', linewidth=3, linestyle='--', alpha=0.8)
# Recovered rotation frame.
ax6.arrow(0, 0, v1_opt[0], v1_opt[1], head_width=0.15, head_length=0.2, fc='tab:green', ec='tab:green', linewidth=3.5, linestyle=':', label='Found')
ax6.arrow(0, 0, v2_opt[0], v2_opt[1], head_width=0.15, head_length=0.2, fc='tab:green', ec='tab:green', linewidth=3.5, linestyle=':')
ax6.set_xlim(-3, 3)
ax6.set_ylim(-3, 3)
ax6.set_xlabel('x₁', fontsize=12)
ax6.set_ylabel('x₂', fontsize=12)
ax6.set_title(f'Comparison: All Frames\nAngle Error = {angle_error:.2f}°', fontsize=14, fontweight='bold')
ax6.legend(fontsize=11, loc='upper left')
ax6.grid(True, alpha=0.3)
ax6.axis('equal')
ax6.axhline(y=0, color='k', linewidth=0.5)
ax6.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

print(f"Determinant: {np.linalg.det(R_opt)}")