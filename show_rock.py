import open3d as o3d
import os

rock_path = "combined_mayan_stone.ply"
if not os.path.exists(rock_path):
    print("Rock not found! Looking for reconstructed_mayan_stone.ply instead...")
    rock_path = "reconstructed_mayan_stone.ply"
    if not os.path.exists(rock_path):
        print("No assembled rock found.")
        exit(1)

print(f"Loading {rock_path} into memory...")
pcd = o3d.io.read_point_cloud(rock_path)

print("Optimizing point cloud resolution for smooth interactive framerates...")
pcd = pcd.voxel_down_sample(0.01)
pcd.estimate_normals()

print("Launching interactive 3D viewer on your screen!")
o3d.visualization.draw_geometries([pcd], window_name="GSoC: Final Assembled Mayan Stele")
