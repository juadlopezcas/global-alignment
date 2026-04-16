import mrcfile
import numpy as np
import os

file_path = 'emd_9515.map' 
recommended_threshold = 0.035 # experiment-Contour_list-level
num_points_wanted = 1000 # downsample

# ==========================================

def map_to_point_cloud(mrc_filename, threshold=None):
    """
    Converts a .map/.mrc density map file into an Nx3 point cloud matrix.
    """
    print(f"Reading file: {mrc_filename} ...")
    
    # 1. Read the density map file
    with mrcfile.open(mrc_filename, mode='r') as mrc:
        # Get the 3D density matrix data
        density_data = mrc.data
        
        # Get the physical size of the voxel
        voxel_size = mrc.voxel_size.x

    print(f"Using provided threshold: {threshold}")

    # 2. Extract voxel index coordinates greater than the threshold
    z_idx, y_idx, x_idx = np.where(density_data > threshold)

    # 3. Combine coordinates into an Nx3 matrix and multiply by voxel size to restore true physical scale
    point_cloud = np.column_stack((x_idx, y_idx, z_idx)) * voxel_size

    return point_cloud

pc = map_to_point_cloud(file_path, threshold=recommended_threshold)

print(f"Extraction complete! Generated {pc.shape[0]} points in total.")
print("The (x, y, z) coordinates of the first 5 points:\n", pc[:5])

# ==========================================

print(f"Original point cloud count: {pc.shape[0]}")

# If the extracted points exceed the desired number, perform random sampling without replacement
if pc.shape[0] > num_points_wanted:
    # Randomly select num_points_wanted indices (replace=False ensures no duplicate points)
    random_indices = np.random.choice(pc.shape[0], num_points_wanted, replace=False)
    
    # Use these indices to extract corresponding points to get the downsampled point cloud
    pc_downsampled = pc[random_indices]
else:
    # If there are fewer points than wanted, do not downsample
    pc_downsampled = pc

print(f"Point cloud count after downsampling: {pc_downsampled.shape[0]}")

# 4. Save the generated point cloud as a .xyz text file for easy input into the MMD algorithm

base_name = os.path.splitext(os.path.basename(file_path))[0]
output_filename = f"{base_name}_{pc_downsampled.shape[0]}.xyz"

np.savetxt(output_filename, pc_downsampled, fmt='%.4f')
print(f"Point cloud file saved as {output_filename}")