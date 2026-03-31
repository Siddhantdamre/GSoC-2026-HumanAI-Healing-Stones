import open3d as o3d
import numpy as np
import os
import glob
import gc
import time
import traceback
import copy
import sys

def preprocess_point_cloud(pcd, voxel_size):
    """Downsamples and computes FPFH features."""
    # Aggressive uniform downsample first to prevent voxel allocation OOM on 500MB files
    pcd_down = pcd.uniform_down_sample(every_k_points=10)
    pcd_down = pcd_down.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
    # --- CURVATURE PRUNING ---
    pcd_down.estimate_covariances(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    covs = np.asarray(pcd_down.covariances)
    
    if len(covs) > 0:
        evals = np.linalg.eigvalsh(covs)
        curvatures = evals[:, 0] / (np.sum(evals, axis=1) + 1e-6)
        
        # Keep only the top 30% jagged edge points
        k = int(len(curvatures) * 0.30)
        edge_indices = np.argsort(curvatures)[-k:]
        pcd_edge = pcd_down.select_by_index(edge_indices)
    else:
        pcd_edge = pcd_down
        
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_edge,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        
    return pcd_down, pcd_edge, pcd_fpfh

def align_pair(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Global RANSAC + Local ICP alignment."""
    distance_threshold = voxel_size * 1.5
    
    # 1. Global RANSAC
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    if result_ransac.fitness < 0.1: # Low quality match
        return None

    # 2. Local ICP Refinement
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size * 0.4, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return result_icp

if __name__ == "__main__":
    # --- Setup ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_ply_files = sorted(glob.glob(os.path.join(base_dir, "*.PLY")) + glob.glob(os.path.join(base_dir, "*.ply")))
    
    # Strictly filter out the massive 4.6GB final assemblies so we don't try to align the full rock as a "fragment"!
    ply_files = []
    for f in all_ply_files:
        basename = os.path.basename(f).lower()
        if "combined" not in basename and "reconstructed" not in basename:
            ply_files.append(f)
            
    if len(ply_files) < 2:
        print("Need at least 2 PLY files for reconstruction.")
        exit(1)

    voxel_size = 0.05  # 5cm resolution to prevent MemoryError on raw 500MB scans
    print(f"\nStarting Global Reconstruction of {len(ply_files)} fragments...")
    
    # --- 1. Pre-process All Fragments ---
    processed_clouds = []
    processed_edges = []
    processed_features = []
    filenames = []

    print("\n[Phase 1/2] Pre-processing all fragments (Extracting curves & features)...")
    for f in ply_files:
        print(f"  - Processing {os.path.basename(f)}...")
        pcd = o3d.io.read_point_cloud(f)
        pcd_down, pcd_edge, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
        processed_clouds.append(pcd_down)
        processed_edges.append(pcd_edge)
        processed_features.append(pcd_fpfh)
        filenames.append(os.path.basename(f))
        del pcd
        gc.collect()

    # --- 2. Iterative Assembly ---
    print("\n[Phase 2/2] Assembling the stone (Iterative Alignment)...")
    
    # Start with the first fragment as our base "stone"
    assembled_indices = [0]
    unaligned_indices = list(range(1, len(processed_clouds)))
    
    # We'll store the transformation from EACH fragment to the BASE coordinate system
    transforms = {0: np.identity(4)}
    
    iteration = 0
    while unaligned_indices and iteration < 50: # Safety break
        iteration += 1
        found_match = False
        
        print(f"\nIteration {iteration}: {len(assembled_indices)} assembled, {len(unaligned_indices)} remaining.")
        
        # Try to match any unaligned piece to ANY piece already in the stone
        for u_idx in unaligned_indices[:]: # Copy list to allow removal
            best_match_result = None
            best_target_idx = -1
            
            # Check against everyone already in the assembly
            for a_idx in assembled_indices:
                print(f"  Testing {filenames[u_idx]} vs {filenames[a_idx]}...", end="\r")
                result = align_pair(
                    processed_edges[u_idx], processed_edges[a_idx],
                    processed_features[u_idx], processed_features[a_idx],
                    voxel_size
                )
                
                if result and (best_match_result is None or result.fitness > best_match_result.fitness):
                    best_match_result = result
                    best_target_idx = a_idx
            
            if best_match_result and best_match_result.fitness > 0.4: # Good enough match
                print(f"  MATCH FOUND! {filenames[u_idx]} snaps to {filenames[best_target_idx]} (Fitness: {best_match_result.fitness:.2f})")
                
                # Calculate transformation to the BASE coordinate system (0)
                # target_to_base = transforms[best_target_idx]
                # source_to_target = best_match_result.transformation
                # source_to_base = target_to_base @ source_to_target
                transforms[u_idx] = transforms[best_target_idx] @ best_match_result.transformation
                
                assembled_indices.append(u_idx)
                unaligned_indices.remove(u_idx)
                found_match = True
                break # Move to next iteration with enlarged stone
        
        if not found_match:
            print("\nCould not find any more matches. Some fragments might not be adjacent yet.")
            break

    # --- 3. Final Merge ---
    print("\nFinalizing Reconstruction...")
    final_stone = o3d.geometry.PointCloud()
    
    for idx in assembled_indices:
        temp_pcd = copy.deepcopy(processed_clouds[idx])
        temp_pcd.transform(transforms[idx])
        final_stone += temp_pcd

    # Optimization
    final_stone = final_stone.voxel_down_sample(voxel_size=0.005)
    final_stone.estimate_normals()
    
    output_path = os.path.join(base_dir, "reconstructed_mayan_stone.ply")
    print(f"Saving final assembly to {output_path}...")
    o3d.io.write_point_cloud(output_path, final_stone, write_ascii=False)
    
    print("\nRECONSTRUCTION COMPLETE!")
    print(f"Assembled {len(assembled_indices)} / {len(ply_files)} fragments.")
    
    # Visualize
    o3d.visualization.draw_geometries([final_stone], window_name="Final Reconstructed Stone")
