import open3d as o3d
import os
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
SHAPENET_PATH = r"C:\Users\RAOVI\PSGN_Project\data\ShapeNetCore.v2"
OUTPUT_PATH = r"C:\Users\RAOVI\PSGN_Project\data\ShapeNet_pointclouds"
N_POINTS = 2048  # number of points per sampled model

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==============================
# FUNCTION: Convert mesh to point cloud
# ==============================
def mesh_to_pointcloud(mesh_path, n_points=2048):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        return None
    mesh.compute_vertex_normals()
    # Uniformly sample points on the surface
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return pcd

# ==============================
# MAIN CONVERSION LOOP
# ==============================
for class_id in tqdm(os.listdir(SHAPENET_PATH), desc="Converting classes"):
    class_dir = os.path.join(SHAPENET_PATH, class_id)
    if not os.path.isdir(class_dir):
        continue

    # Create same folder structure in output path
    output_class_dir = os.path.join(OUTPUT_PATH, class_id)
    os.makedirs(output_class_dir, exist_ok=True)

    # Loop through all models inside each class
    for model_id in os.listdir(class_dir):
        model_dir = os.path.join(class_dir, model_id, "models")
        obj_file = os.path.join(model_dir, "model_normalized.obj")

        if not os.path.exists(obj_file):
            continue

        try:
            # Convert mesh to point cloud
            pcd = mesh_to_pointcloud(obj_file, N_POINTS)
            if pcd is None:
                continue

            # Save file using model_id as name, e.g. 1a04e3eab45ca15dd86060f189eb133.ply
            save_path = os.path.join(output_class_dir, f"{model_id}.ply")
            o3d.io.write_point_cloud(save_path, pcd)

        except Exception as e:
            print(f"⚠️ Error converting {obj_file}: {e}")

print("\n✅ Conversion complete — all meshes saved as point clouds in ShapeNet_pointclouds.")
