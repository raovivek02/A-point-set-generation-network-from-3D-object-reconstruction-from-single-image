import torch
import torch.nn as nn
import numpy as np

class PSGN(nn.Module):
    def __init__(self, num_points=2048):
        super(PSGN, self).__init__()
        # -------- Encoder (Image feature extractor) --------
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 5, stride=2, padding=2), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_feat = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True)
        )

        # -------- Coarse Point Generator --------
        self.coarse = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, num_points // 2 * 3)
        )

        # -------- Fine Point Generator (offset decoder) --------
        self.fine = nn.Sequential(
            nn.Linear(1024 + 2, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 3)
        )

        self.num_points = num_points

    def forward(self, x):
        B = x.size(0)
        feat = self.encoder(x).view(B, -1)
        feat = self.fc_feat(feat)

        # Coarse output
        coarse = self.coarse(feat).view(B, self.num_points // 2, 3)

        # Fine output grid
        grid_size = int(np.sqrt(self.num_points // 2))
        u = torch.linspace(-0.05, 0.05, grid_size, device=x.device)
        v = torch.linspace(-0.05, 0.05, grid_size, device=x.device)
        uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).view(-1, 2)
        uv = uv.unsqueeze(0).repeat(B, 1, 1)

        feat_expand = feat.unsqueeze(1).repeat(1, uv.size(1), 1)
        fine_input = torch.cat([uv, feat_expand], dim=-1)
        offset = self.fine(fine_input)
        fine = coarse + offset

        return coarse, fine

