"""
GSoC 2026 - Healing Stones: Fragment Alignment
================================================
3-tier resolution approach:
  Tier 1 (coarse): 10cm voxel for fast RANSAC (~15K-30K pts)  
  Tier 2 (medium): 5cm voxel for ICP refinement (~50K-100K pts)
  Tier 3 (fine):   2cm voxel for display (~200K+ pts)
"""

import open3d as o3d
import numpy as np
import os
import sys
import gc


def load_downsample(filepath, every_k, voxel):
    """Load PLY, decimate, voxel grid, estimate normals."""
    pcd = o3d.io.read_point_cloud(filepath)
    n_raw = len(pcd.points)
    pcd = pcd.uniform_down_sample(every_k_points=every_k)
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    print(f"    {n_raw:,} -> {len(pcd.points):,} pts (k={every_k}, v={voxel})")
    return pcd


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) == 3:
        file1, file2 = sys.argv[1], sys.argv[2]
        if not os.path.isabs(file1): file1 = os.path.join(base_dir, file1)
        if not os.path.isabs(file2): file2 = os.path.join(base_dir, file2)
    else:
        file1 = os.path.join(base_dir, "NAR_ST_43B_FR_01_F_01_R_02.PLY")
        file2 = os.path.join(base_dir, "NAR_ST_43B_FR_02_F_01_R_01.PLY")

    for f in [file1, file2]:
        if not os.path.exists(f):
            print(f"Error: {f} not found"); sys.exit(1)

    n1, n2 = os.path.basename(file1), os.path.basename(file2)
    print("=" * 60)
    print("  Healing Stones - Geometric Fragment Alignment")
    print("=" * 60)
    print(f"  Source: {n1}")
    print(f"  Target: {n2}")

    # ===== TIER 1: Very coarse clouds for fast RANSAC =====
    print(f"\n[1/6] Loading coarse clouds for RANSAC (15cm voxel)...")
    print(f"  Source:")
    src_coarse = load_downsample(file1, every_k=200, voxel=0.15)
    gc.collect()
    print(f"  Target:")
    tgt_coarse = load_downsample(file2, every_k=200, voxel=0.15)
    gc.collect()

    print(f"\n[2/6] Computing FPFH features...")
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_coarse, o3d.geometry.KDTreeSearchParamHybrid(radius=0.75, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_coarse, o3d.geometry.KDTreeSearchParamHybrid(radius=0.75, max_nn=100))
    print(f"  Source FPFH shape: {src_fpfh.data.shape}")
    print(f"  Target FPFH shape: {tgt_fpfh.data.shape}")

    print(f"\n[3/6] RANSAC global registration (100K iterations)...")
    dist = 0.20  # 20cm threshold for coarse matching
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_coarse, tgt_coarse, src_fpfh, tgt_fpfh, True, dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    print(f"  Fitness: {result_ransac.fitness:.4f}")
    print(f"  RMSE:    {result_ransac.inlier_rmse:.6f}")
    print(f"  Correspondences: {len(result_ransac.correspondence_set)}")

    del src_coarse, tgt_coarse, src_fpfh, tgt_fpfh
    gc.collect()

    # ===== TIER 2: Medium clouds for ICP refinement =====
    print(f"\n[4/6] Loading medium clouds for ICP (5cm voxel)...")
    print(f"  Source:")
    src_med = load_downsample(file1, every_k=50, voxel=0.05)
    gc.collect()
    print(f"  Target:")
    tgt_med = load_downsample(file2, every_k=50, voxel=0.05)
    gc.collect()

    print(f"\n[5/6] ICP Point-to-Plane refinement...")
    result_icp = o3d.pipelines.registration.registration_icp(
        src_med, tgt_med, 0.025, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    print(f"  Fitness: {result_icp.fitness:.4f}")
    print(f"  RMSE:    {result_icp.inlier_rmse:.6f}")
    print(f"\n  Final Matrix:\n{result_icp.transformation}")

    final_T = result_icp.transformation
    del src_med, tgt_med
    gc.collect()

    # ===== TIER 3: Fine clouds for display =====
    print(f"\n[6/6] Loading display meshes (2cm voxel)...")
    print(f"  Source:")
    vis_src = load_downsample(file1, every_k=50, voxel=0.02)
    gc.collect()
    print(f"  Target:")
    vis_tgt = load_downsample(file2, every_k=50, voxel=0.02)
    gc.collect()

    vis_src.transform(final_T)

    print(f"\n  Opening 3D viewer...")
    o3d.visualization.draw_geometries(
        [vis_src, vis_tgt],
        window_name="Mayan Stone Fragment Assembly - GSoC Healing Stones",
        width=1280, height=720)

    print("Done.")
