import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from models.psgn_model import PSGN

# ============================================================
# ✅ CONFIGURATION
# ============================================================
MODEL_PATH = r"C:\Users\RAOVI\PSGN_Project\outputs\psgn_epoch_10.pth"
INPUT_PATH = r"C:\Users\RAOVI\PSGN_Project\inputs\plane1.png"
NUM_POINTS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ✅ LOAD MODEL
# ============================================================
model = PSGN(num_points=NUM_POINTS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully.")

# ============================================================
# ✅ IMAGE PREPROCESSING
# ============================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

img = Image.open(INPUT_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)
print("🖼️ Loaded input image:", INPUT_PATH)

# ============================================================
# ✅ INFERENCE
# ============================================================
with torch.no_grad():
    coarse, fine = model(img_tensor)
    coarse = coarse.squeeze(0).cpu().numpy()
    fine = fine.squeeze(0).cpu().numpy()

# ============================================================
# ✅ VISUALIZATION
# ============================================================
fig = plt.figure(figsize=(12, 4))

# Input
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img)
ax1.axis("off")
ax1.set_title("Input 2D Image")

# Coarse Output
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.scatter(coarse[:, 0], coarse[:, 1], coarse[:, 2], c='gray', s=10)
ax2.set_title("Coarse 3D Point Cloud")

# Fine Output
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.scatter(fine[:, 0], fine[:, 1], fine[:, 2], c='lightgray', s=8)
ax3.set_title("Fine 3D Point Cloud")

plt.tight_layout()
plt.show()

# ============================================================
# ✅ SAVE BOTH STAGES AS .PLY FILES
# ============================================================
os.makedirs("outputs", exist_ok=True)
coarse_ply = "outputs/plane1_coarse.ply"
fine_ply = "outputs/plane1_fine.ply"

for arr, path in [(coarse, coarse_ply), (fine, fine_ply)]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    o3d.io.write_point_cloud(path, pcd)

print("💾 Saved coarse & fine point clouds.")
