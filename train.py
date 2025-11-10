import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import open3d as o3d
from tqdm import tqdm

# ============================================================
# ✅ CONFIGURATION
# ============================================================
images_root = r"C:\Users\RAOVI\PSGN_Project\data\renders\rendering"
pointcloud_root = r"C:\Users\RAOVI\PSGN_Project\data\ShapeNet_pointclouds"
save_path = "outputs/psgn_epoch_10.pth"
batch_size = 8
epochs = 10
num_points = 2048
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ✅ CHAMFER DISTANCE LOSS
# ============================================================
def chamfer_distance(p1, p2):
    x, y = p1.unsqueeze(2), p2.unsqueeze(1)  # (B, N, 1, 3), (B, 1, M, 3)
    dist = torch.norm(x - y, dim=-1)         # (B, N, M)
    dist1, _ = torch.min(dist, dim=2)
    dist2, _ = torch.min(dist, dim=1)
    return (dist1.mean() + dist2.mean())

# ============================================================
# ✅ PSGN MODEL (Encoder + Coarse + Fine)
# ============================================================
class PSGN(nn.Module):
    def __init__(self, num_points=2048):
        super(PSGN, self).__init__()
        # Encoder (as in PSGN paper, CNN → 1024-d feature)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 5, stride=2, padding=2), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_feat = nn.Sequential(nn.Linear(512, 1024), nn.ReLU())

        # Coarse Point Generator
        self.coarse = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points // 2 * 3)
        )

        # Fine Point Refinement (folding-style)
        self.fine = nn.Sequential(
            nn.Linear(1024 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        self.num_points = num_points

    def forward(self, x):
        B = x.size(0)
        feat = self.encoder(x).view(B, -1)
        feat = self.fc_feat(feat)

        # Coarse point cloud (N/2 x 3)
        coarse = self.coarse(feat).view(B, self.num_points // 2, 3)

        # Generate local folding grid
        grid_size = int(np.sqrt(self.num_points // 2))
        u = torch.linspace(-0.05, 0.05, grid_size, device=x.device)
        v = torch.linspace(-0.05, 0.05, grid_size, device=x.device)
        uv = torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1).view(-1, 2)
        uv = uv.unsqueeze(0).repeat(B, 1, 1)

        # Repeat global feature
        feat_expand = feat.unsqueeze(1).repeat(1, uv.size(1), 1)
        fine_input = torch.cat([uv, feat_expand], dim=-1)

        offsets = self.fine(fine_input)
        fine = coarse + offsets[:, : coarse.size(1), :]

        return fine

# ============================================================
# ✅ DATASET LOADER
# ============================================================
class ShapeNetPairDataset(Dataset):
    def __init__(self, images_root, pc_root, transform=None):
        self.samples = []
        self.transform = transform

        for cls in os.listdir(images_root):
            img_cls_dir = os.path.join(images_root, cls)
            pc_cls_dir = os.path.join(pc_root, cls)
            if not os.path.isdir(img_cls_dir) or not os.path.isdir(pc_cls_dir):
                continue

            for inst in os.listdir(img_cls_dir):
                inst_path = os.path.join(img_cls_dir, inst)
                if not os.path.isdir(inst_path):
                    continue

                # ✅ Handle instance names with class prefix
                inst_clean = inst.split("_")[-1] if "_" in inst else inst
                pc_path = os.path.join(pc_cls_dir, f"{inst_clean}.ply")

                if not os.path.isfile(pc_path):
                    # fallback
                    alt = os.path.join(pc_cls_dir, f"{inst}.ply")
                    if os.path.isfile(alt):
                        pc_path = alt
                    else:
                        continue

                img_files = [f for f in os.listdir(inst_path) if f.endswith(".png")]
                if not img_files:
                    continue

                # pick first rendering (front view)
                img_path = os.path.join(inst_path, img_files[0])
                self.samples.append((img_path, pc_path))

        print(f"✅ Loaded {len(self.samples)} paired samples from images_root={images_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pc_path = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Load point cloud
        pcd = o3d.io.read_point_cloud(pc_path)
        points = np.asarray(pcd.points, dtype=np.float32)

        # Normalize point cloud
        points = points - np.mean(points, axis=0)
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / scale

        # Randomly sample num_points
        if points.shape[0] > num_points:
            choice = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[choice, :]

        return img, torch.from_numpy(points)

# ============================================================
# ✅ TRAINING LOOP
# ============================================================
def train():
    dataset = ShapeNetPairDataset(images_root, pointcloud_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = PSGN(num_points=num_points).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Using device: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for img, gt_points in pbar:
            img, gt_points = img.to(device), gt_points.to(device)

            pred_points = model(img)
            loss = chamfer_distance(pred_points, gt_points)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.6f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved at {save_path}")


if __name__ == "__main__":
    train()

