import open3d as o3d
import os
import glob
import gc
import traceback
import sys

def process_and_view(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return None

        print(f"Loading {os.path.basename(file_path)}...")
        pcd = o3d.io.read_point_cloud(file_path)
        
        if pcd.is_empty():
            print(f"  - Warning: File is empty or invalid.")
            return None

        # Initial downsampling for memory efficiency
        print(f"  - Downsampling (1.5cm resolution)...")
        down_pcd = pcd.voxel_down_sample(voxel_size=0.015)
        
        del pcd
        gc.collect()
        
        return down_pcd
    except Exception as e:
        print(f"  - FAILED: {e}")
        return None

if __name__ == "__main__":
    try:
        # Standard path resolution
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ply_files = glob.glob(os.path.join(base_dir, "*.PLY"))
        
        if not ply_files:
            ply_files = glob.glob("*.PLY")
            if not ply_files:
                print(f"No .PLY files found.")
                sys.exit(1)
            else:
                base_dir = os.getcwd()

        print(f"\nProcessing {len(ply_files)} fragments...")
        
        clouds = []
        for i, f in enumerate(ply_files):
            print(f"[{i+1}/{len(ply_files)}] ", end="")
            pcd = process_and_view(f)
            if pcd:
                clouds.append(pcd)
        
        if not clouds:
            print("No point clouds loaded.")
            sys.exit(1)

        # Combine fragments
        print(f"\nCombining {len(clouds)} fragments...")
        combined_pcd = o3d.geometry.PointCloud()
        for c in clouds:
            combined_pcd += c
        
        # FINAL OPTIMIZATIONS
        print(":: Running final optimizations...")
        # 1. Final downsampling to keep file size small
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005)
        
        # 2. Unify normals for better lighting (prevents blowout)
        print(":: Re-estimating unified normals...")
        combined_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # 3. Save as BINARY PLY (much smaller than ASCII)
        output_path = os.path.join(base_dir, "combined_mayan_stone.ply")
        print(f":: Saving BINARY PLY to {output_path}...")
        # write_ascii=False is the default, but we'll be explicit for clarity
        o3d.io.write_point_cloud(output_path, combined_pcd, write_ascii=False)
        
        print(f"\nSUCCESS! Results saved and optimized.")
        print("Opening 3D viewer. Use mouse to rotate.")
        
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries(clouds + [coord], window_name="Mayan Stone Reconstructor - Optimized")

    except Exception as e:
        print(f"\nCRITICAL ERROR:")
        traceback.print_exc()
        input("\nPress Enter to close...")