import torch
import open3d as o3d
from models.psgn_model import PSGN
from utils.dataset_loader import ChairDataset
from models.losses import chamfer_distance

# ========================
# Load model
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PSGN(num_points=2048).to(device)
model.load_state_dict(torch.load("outputs/psgn_chair.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully")

# ========================
# Load one sample
# ========================
dataset = ChairDataset(r"data\shape_net_core_uniform_samples_2048\03001627")
img, gt_points = dataset[0]
img, gt_points = img.unsqueeze(0).to(device), gt_points.unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    pred_points = model(img)

# Compute Chamfer Distance
loss = chamfer_distance(pred_points, gt_points)
print(f"Chamfer Distance (Reconstruction Error): {loss.item():.6f}")

# ========================
# Convert tensors to Open3D format
# ========================
pred_np = pred_points.squeeze(0).cpu().numpy()
gt_np = gt_points.squeeze(0).cpu().numpy()

pred_pcd = o3d.geometry.PointCloud()
pred_pcd.points = o3d.utility.Vector3dVector(pred_np)

gt_pcd = o3d.geometry.PointCloud()
gt_pcd.points = o3d.utility.Vector3dVector(gt_np)

pred_pcd.paint_uniform_color([1, 0, 0])   # red: predicted
gt_pcd.paint_uniform_color([0, 1, 0])     # green: ground truth

# ========================
# Visualize both
# ========================
o3d.visualization.draw_geometries([pred_pcd.translate((0.5,0,0)), gt_pcd.translate((-0.5,0,0))])
