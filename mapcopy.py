import mrcfile
import numpy as np
import os
import argparse

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

def random_downsample(pc, num_points):
    """
    Helper function to randomly downsample a point cloud.
    """
    if pc.shape[0] > num_points:
        random_indices = np.random.choice(pc.shape[0], num_points, replace=False)
        return pc[random_indices]
    return pc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Map to Dual Point Clouds (X and Y)")
    parser.add_argument('--file', type=str, default='emd_9515.map', help='Path to the .map/.mrc file')
    parser.add_argument('--threshold', type=float, default=0.035, help='Density threshold')
    parser.add_argument('--num_x', type=int, default=1000, help='Number of target points for point cloud X')
    parser.add_argument('--num_y', type=int, default=700, help='Number of target points for point cloud Y')
    
    args = parser.parse_args()

    pc = map_to_point_cloud(args.file, threshold=args.threshold)
    print(f"Extraction complete! Generated {pc.shape[0]} points in total from the density map.\n")


    print(f"Generating independent samples...")
    pc_X = random_downsample(pc, args.num_x)
    pc_Y = random_downsample(pc, args.num_y)

    print(f"  -> Point cloud X count: {pc_X.shape[0]}")
    print(f"  -> Point cloud Y count: {pc_Y.shape[0]}\n")

    base_name = os.path.splitext(os.path.basename(args.file))[0]
    
    output_x = f"{base_name}_X_{pc_X.shape[0]}.xyz"
    output_y = f"{base_name}_Y_{pc_Y.shape[0]}.xyz"

    np.savetxt(output_x, pc_X, fmt='%.4f')
    np.savetxt(output_y, pc_Y, fmt='%.4f')

    print(f"Successfully saved:")
    print(f"  [X] {output_x}")
    print(f"  [Y] {output_y}")