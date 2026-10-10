"""
Train a U-Net-style convolutional VAE on MNIST.

Outputs:
  - checkpoints/mnist_unet_vae.pt
  - outputs/recon_epoch_XXX.png
  - outputs/sample_epoch_XXX.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class UNetVAE(nn.Module):
    """A compact U-Net-style VAE for 28x28 grayscale images."""

    def __init__(self, latent_dim: int = 32, base: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base = base

        # Encoder: 1x28x28 -> base*4 x 7 x 7
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
        )

        flat_dim = base * 4 * 7 * 7
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder: latent -> base*4 x 7 x 7 -> 1x28x28
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.dec_mid = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
        )
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.mid(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), self.base * 4, 7, 7)
        h = self.dec_mid(h)
        h = self.dec1(h)
        h = self.dec2(h)
        return self.out(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar


parser = argparse.ArgumentParser(description="Train a U-Net-style VAE on MNIST.")
parser.add_argument("--data-dir", type=Path, default=Path("data"))
parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--latent-dim", type=int, default=32)
parser.add_argument("--base", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--warmup-epochs", type=int, default=5)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.out_dir.mkdir(parents=True, exist_ok=True)
args.ckpt_dir.mkdir(parents=True, exist_ok=True)

# Data: MNIST pixels are in [0, 1], matching Bernoulli likelihood.
transform = transforms.ToTensor()
train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=device.type == "cuda",
)

model = UNetVAE(latent_dim=args.latent_dim, base=args.base).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
fixed_noise = torch.randn(64, args.latent_dim, device=device)

for epoch in range(1, args.epochs + 1):
    model.train()
    beta = min(1.0, epoch / max(1, args.warmup_epochs)) * args.beta
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for x, _ in train_loader:
        x = x.to(device)
        recon_logits, mu, logvar = model(x)

        recon_loss = F.binary_cross_entropy_with_logits(
            recon_logits,
            x,
            reduction="sum",
        ) / x.size(0)
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / x.size(0)
        loss = recon_loss + beta * kl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_recon += recon_loss.item() * x.size(0)
        total_kl += kl_loss.item() * x.size(0)

    n = len(train_loader.dataset)
    print(
        f"Epoch {epoch:03d}/{args.epochs} | "
        f"beta={beta:.3f} | "
        f"loss={total_loss / n:.3f} | "
        f"recon={total_recon / n:.3f} | "
        f"kl={total_kl / n:.3f}"
    )

    # Save reconstruction and prior samples for visual inspection.
    model.eval()
    with torch.no_grad():
        x_vis, _ = next(iter(train_loader))
        x_vis = x_vis[:8].to(device)
        recon_vis = torch.sigmoid(model(x_vis)[0])
        recon_grid = torch.cat([x_vis, recon_vis], dim=0)
        utils.save_image(
            recon_grid,
            args.out_dir / f"recon_epoch_{epoch:03d}.png",
            nrow=8,
        )

        sample = torch.sigmoid(model.decode(fixed_noise))
        utils.save_image(
            sample,
            args.out_dir / f"sample_epoch_{epoch:03d}.png",
            nrow=8,
        )

checkpoint = {
    "model_state": model.state_dict(),
    "latent_dim": args.latent_dim,
    "base": args.base,
}
torch.save(checkpoint, args.ckpt_dir / "mnist_unet_vae.pt")
print(f"Saved checkpoint to {args.ckpt_dir / 'mnist_unet_vae.pt'}")
