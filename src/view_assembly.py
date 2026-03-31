"""
GSoC 2026 - Healing Stones: Multi-Fragment Assembly Viewer
============================================================
Loads multiple Mayan stone fragments and displays them together.
If the fragments were scanned in a shared coordinate system,
they will appear in their correct relative positions.
"""

import open3d as o3d
import numpy as np
import os
import sys
import gc
import glob


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_ply = sorted(glob.glob(os.path.join(base_dir, "NAR_ST_43B_FR_*.PLY")))

    if not all_ply:
        print("No PLY fragment files found.")
        sys.exit(1)

    print("=" * 60)
    print("  Healing Stones - Multi-Fragment Assembly Viewer")
    print("=" * 60)
    print(f"  Found {len(all_ply)} fragments\n")

    clouds = []
    # Assign different colors to each fragment for visual distinction
    colors = [
        [0.9, 0.3, 0.2],  # Red
        [0.2, 0.6, 0.9],  # Blue
        [0.3, 0.8, 0.3],  # Green
        [0.9, 0.7, 0.1],  # Yellow
        [0.7, 0.3, 0.8],  # Purple
        [0.9, 0.5, 0.2],  # Orange
        [0.2, 0.8, 0.7],  # Teal
        [0.8, 0.4, 0.5],  # Pink
        [0.5, 0.5, 0.9],  # Periwinkle
        [0.6, 0.8, 0.2],  # Lime
        [0.4, 0.3, 0.6],  # Indigo
        [0.9, 0.6, 0.6],  # Salmon
        [0.3, 0.5, 0.4],  # Forest
        [0.7, 0.7, 0.3],  # Olive
        [0.5, 0.2, 0.3],  # Maroon
        [0.3, 0.7, 0.6],  # Jade
        [0.8, 0.3, 0.6],  # Magenta
    ]

    for i, f in enumerate(all_ply):
        name = os.path.basename(f)
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"  [{i+1}/{len(all_ply)}] Loading {name} ({size_mb:.0f} MB)...", end=" ")

        pcd = o3d.io.read_point_cloud(f)
        # Aggressive downsample to fit all fragments in memory
        pcd = pcd.uniform_down_sample(every_k_points=100)
        pcd = pcd.voxel_down_sample(0.03)

        # Color each fragment differently
        color = colors[i % len(colors)]
        pcd.paint_uniform_color(color)

        print(f"{len(pcd.points):,} pts")
        clouds.append(pcd)
        gc.collect()

    print(f"\n  Total clouds: {len(clouds)}")
    print(f"  Total display points: {sum(len(c.points) for c in clouds):,}")
    print(f"\n  Opening 3D Viewer (each fragment has a different color)...")

    o3d.visualization.draw_geometries(
        clouds,
        window_name="Mayan Stele - All Fragments Assembly",
        width=1280, height=720)

    print("Done.")
