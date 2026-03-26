import open3d as o3d
import numpy as np
import copy
import os
import sys

def preprocess_point_cloud(pcd, voxel_size):
    """Downsamples the cloud and computes geometric features for matching."""
    print(f":: Preprocessing (Voxel: {voxel_size})...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Uses RANSAC to find a rough alignment between the two pieces."""
    print(":: Running RANSAC Global Registration...")
    distance_threshold = voxel_size * 1.5
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result

def refine_registration(source, target, global_transform, voxel_size):
    """Uses Iterative Closest Point (ICP) to snap the pieces perfectly together."""
    print(":: Running ICP Local Refinement...")
    distance_threshold = voxel_size * 0.4
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, global_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return result

def draw_registration_result(source, target, transformation):
    """Visualizes the aligned fragments in Cyan and Yellow."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.paint_uniform_color([1, 0.706, 0]) # Yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # Cyan
    
    source_temp.transform(transformation)
    
    o3d.visualization.draw_geometries([source_temp, target_temp], 
                                      window_name="GSoC Healing Stones - Alignment")

if __name__ == "__main__":
    # Path resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Files to align (hardcoded for test piece 01 and 02)
    file1 = os.path.join(base_dir, "NAR_ST_43B_FR_01_F_01_R_02.PLY")
    file2 = os.path.join(base_dir, "NAR_ST_43B_FR_02_F_01_R_01.PLY")
    
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("Error: Fragment files not found in the root directory.")
        sys.exit(1)
        
    print(f"Aligning:\n  1: {os.path.basename(file1)}\n  2: {os.path.basename(file2)}")
    
    source = o3d.io.read_point_cloud(file1)
    target = o3d.io.read_point_cloud(file2)
    
    # Adjust voxel_size to 0.01 (1cm) if it's too slow
    voxel_size = 0.005 
    
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    global_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    
    if global_result.fitness > 0:
        final_result = refine_registration(source_down, target_down, global_result.transformation, voxel_size)
        print(f"\nFinal alignment matrix:\n{final_result.transformation}")
        draw_registration_result(source_down, target_down, final_result.transformation)
    else:
        print("\nGlobal registration failed to find a match. Pieces might not be adjacent.")
