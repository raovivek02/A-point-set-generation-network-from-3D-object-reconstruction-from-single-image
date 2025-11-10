import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from config.data_config import get_paired_data
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes

class PSGNDataset(Dataset):
    def __init__(self, transform=None, num_points=2048):
        self.data_pairs = get_paired_data()
        self.transform = transform
        self.num_points = num_points

    def __len__(self):
        return len(self.data_pairs)

    def load_point_cloud(self, obj_path):
        # Load OBJ file and sample point cloud
        mesh = load_objs_as_meshes([obj_path])
        verts = mesh.verts_list()[0].cpu().numpy()

        # Randomly sample points to a fixed size
        if verts.shape[0] > self.num_points:
            idx = np.random.choice(verts.shape[0], self.num_points, replace=False)
        else:
            idx = np.random.choice(verts.shape[0], self.num_points, replace=True)
        return verts[idx]

    def __getitem__(self, idx):
        img_path, obj_path = self.data_pairs[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = np.asarray(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img.transpose(2, 0, 1))

        # Load point cloud
        point_cloud = self.load_point_cloud(obj_path)
        point_cloud = torch.from_numpy(point_cloud).float()

        return img, point_cloud

if __name__ == "__main__":
    dataset = PSGNDataset()
    print(f"✅ Total samples: {len(dataset)}")

    img, cloud = dataset[0]
    print("🖼 Image tensor:", img.shape)
    print("🌐 Point cloud tensor:", cloud.shape)
