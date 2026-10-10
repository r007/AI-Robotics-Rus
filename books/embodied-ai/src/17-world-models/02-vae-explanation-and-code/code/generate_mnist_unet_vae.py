"""
Generate MNIST-like images from a trained U-Net-style VAE checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import utils


class UNetVAE(nn.Module):
    """Same architecture as the training script, kept self-contained."""

    def __init__(self, latent_dim: int = 32, base: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base = base

        self.enc1 = nn.Sequential(nn.Conv2d(1, base, 3, padding=1), nn.GroupNorm(4, base), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU())
        self.mid = nn.Sequential(nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU())
        flat_dim = base * 4 * 7 * 7
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.dec_mid = nn.Sequential(nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.GroupNorm(4, base), nn.SiLU())
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), self.base * 4, 7, 7)
        h = self.dec_mid(h)
        h = self.dec1(h)
        h = self.dec2(h)
        return self.out(h)


parser = argparse.ArgumentParser(description="Generate MNIST-like images with a trained VAE.")
parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mnist_unet_vae.pt"))
parser.add_argument("--out", type=Path, default=Path("outputs/generated.png"))
parser.add_argument("--num-samples", type=int, default=64)
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.out.parent.mkdir(parents=True, exist_ok=True)

checkpoint = torch.load(args.checkpoint, map_location=device)
model = UNetVAE(
    latent_dim=int(checkpoint["latent_dim"]),
    base=int(checkpoint["base"]),
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

with torch.no_grad():
    z = torch.randn(args.num_samples, model.latent_dim, device=device)
    images = torch.sigmoid(model.decode(z))
    utils.save_image(images, args.out, nrow=int(args.num_samples ** 0.5))

print(f"Saved generated images to {args.out}")
