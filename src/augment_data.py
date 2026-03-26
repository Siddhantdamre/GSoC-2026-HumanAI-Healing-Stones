import open3d as o3d
import numpy as np
import os
import random
import json

def get_random_plane(pcd):
    """Generates a random cutting plane passing through the bounding box."""
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    
    # Random point near the center
    offset = (np.random.rand(3) - 0.5) * extent * 0.5
    point_on_plane = center + offset
    
    # Random normal vector
    normal = np.random.randn(3)
    normal /= np.linalg.norm(normal)
    
    return point_on_plane, normal

def split_point_cloud(pcd, point, normal):
    """Splits a point cloud into two using a plane."""
    points = np.asarray(pcd.points)
    vectors = points - point
    # Dot product with normal to find which side of the plane
    dots = np.dot(vectors, normal)
    
    idx_positive = np.where(dots > 0)[0]
    idx_negative = np.where(dots <= 0)[0]
    
    pcd_pos = pcd.select_by_index(idx_positive)
    pcd_neg = pcd.select_by_index(idx_negative)
    
    return pcd_pos, pcd_neg

def generate_random_transform():
    """Generates a random SE(3) transformation matrix."""
    # Random translation (-0.5 to 0.5 coordinate units)
    t = (np.random.rand(3) - 0.5) * 1.0
    
    # Random rotation (Euler angles to rotation matrix)
    angles = np.random.rand(3) * 2 * np.pi
    R = o3d.geometry.get_rotation_matrix_from_xyz(angles)
    
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

if __name__ == "__main__":
    # --- Config ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # We will pretend the reconstructed stone is our "perfect" unbroken artifact
    input_file = os.path.join(base_dir, "reconstructed_mayan_stone.ply")
    output_dir = os.path.join(base_dir, "training_data", "synthetic_fractures")
    num_cuts = 3 # 3 cuts = roughly 2^3 = 8 pieces
    
    if not os.path.exists(input_file):
        print(f"Warning: Full reconstructed stone not found at {input_file}.")
        print("Using Fragment 01 as a fallback base to demonstrate the ML pipeline...")
        input_file = os.path.join(base_dir, "NAR_ST_43B_FR_01_F_01_R_02.PLY")
        if not os.path.exists(input_file):
            print("Error: No valid .PLY files found to shatter.")
            exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading Base Artifact: {input_file}")
    base_pcd = o3d.io.read_point_cloud(input_file)
    
    # --- Shattering ---
    print(f"Shattering model with {num_cuts} planar cuts...")
    fragments = [base_pcd]
    
    for i in range(num_cuts):
        new_fragments = []
        for frag in fragments:
            if len(frag.points) < 1000: # Don't cut tiny dust pieces
                new_fragments.append(frag)
                continue
            pt, n = get_random_plane(frag)
            pcd1, pcd2 = split_point_cloud(frag, pt, n)
            if len(pcd1.points) > 0: new_fragments.append(pcd1)
            if len(pcd2.points) > 0: new_fragments.append(pcd2)
        fragments = new_fragments

    print(f"Fractured the artifact into {len(fragments)} chunks.")
    
    # --- Scrambling & Saving ---
    print("Scrambling fragments (Data Augmentation) and calculating Ground Truth labels...")
    ground_truth = {}
    
    for i, frag in enumerate(fragments):
        # 1. Generate random transform
        T = generate_random_transform()
        
        # 2. Transform the fragment (simulating randomized puzzle pieces)
        frag.transform(T)
        
        # 3. Save fragment
        frag_name = f"fragment_{i:03d}.ply"
        o3d.io.write_point_cloud(os.path.join(output_dir, frag_name), frag, write_ascii=False)
        
        # 4. Save the inverse transform! 
        # ML models train by trying to predict the transform that undoes the scramble.
        # Inverse of T brings the piece back to its original location.
        T_inv = np.linalg.inv(T)
        ground_truth[frag_name] = T_inv.tolist()

    # Save ground truth to JSON for the ML Model to train on
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"\nSUCCESS! ML Training Dataset generated in:\n{output_dir}")
    print("This pipeline successfully automates data augmentation as requested by CERN.")
