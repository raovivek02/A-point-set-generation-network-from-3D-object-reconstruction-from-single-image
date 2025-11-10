import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import open3d as o3d


class ShapeNetPairDataset(Dataset):
    """
    Loads paired 2D rendered images and 3D point clouds.
    Compatible with dataset structure:
    renders/rendering/<class>/<class>_<model_id>/0.png
    ShapeNet_pointclouds/<class>/<model_id>.ply
    """

    def __init__(self, render_root, pc_root, class_ids=None, transform=None, num_points=2048):
        """
        Args:
            render_root (str): Path to renders directory
                e.g., data/renders/rendering
            pc_root (str): Path to point cloud directory
                e.g., data/ShapeNet_pointclouds
            class_ids (list): List of class IDs (e.g., ['02691156', '02958343', '03001627'])
            transform: Optional image transform
            num_points (int): Number of points to sample from each point cloud
        """
        self.render_root = render_root
        self.pc_root = pc_root
        self.class_ids = class_ids or ['02691156', '02958343', '03001627']  # airplane, car, chair
        self.transform = transform
        self.num_points = num_points
        self.pairs = []

        print("🔍 Scanning dataset folders for valid (image, point cloud) pairs...")

        for cls in self.class_ids:
            render_cls_dir = os.path.join(self.render_root, cls)
            pc_cls_dir = os.path.join(self.pc_root, cls)

            if not os.path.exists(render_cls_dir) or not os.path.exists(pc_cls_dir):
                print(f"⚠️ Skipping {cls}: folder missing.")
                continue

            for folder in os.listdir(render_cls_dir):
                model_id = folder.split(f"{cls}_")[-1]  # extract ID part
                img_path = os.path.join(render_cls_dir, folder, "0.png")
                ply_path = os.path.join(pc_cls_dir, f"{model_id}.ply")

                if os.path.exists(img_path) and os.path.exists(ply_path):
                    self.pairs.append((img_path, ply_path))

        print(f"✅ Found {len(self.pairs)} valid image–point cloud pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, ply_path = self.pairs[idx]

        # --- Load image ---
        img = Image.open(img_path).convert("RGB").resize((128, 128))
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        # --- Load point cloud ---
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points, dtype=np.float32)

        # --- Resample or repeat points ---
        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice, :]

        # --- Normalize point cloud ---
        points -= np.mean(points, axis=0)
        points /= np.max(np.linalg.norm(points, axis=1))

        return img, torch.from_numpy(points)


# ======================================================
# ✅ TEST SCRIPT (run this to verify your dataset)
# ======================================================
if __name__ == "__main__":
    render_root = r"C:\Users\RAOVI\PSGN_Project\data\renders\rendering"
    pc_root = r"C:\Users\RAOVI\PSGN_Project\data\ShapeNet_pointclouds"

    dataset = ShapeNetPairDataset(render_root, pc_root)
    print(f"Dataset size: {len(dataset)} samples")

    img, pc = dataset[0]
    print(f"🖼️ Image shape: {img.shape}")
    print(f"🌀 Point cloud shape: {pc.shape}")
