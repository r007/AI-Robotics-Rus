"""
Train a compact DDPM on labeled STL10 images.

Outputs:
  - checkpoints/stl10_ddpm.pt
  - outputs/stl10_sample_epoch_XXX.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = t.float()[:, None] * freq[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return F.pad(emb, (0, self.dim % 2))


def norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups(out_ch), out_ch)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet96(nn.Module):
    """A small U-Net for 96x96 RGB DDPM noise prediction."""

    def __init__(self, in_ch: int = 3, base: int = 48, time_dim: int = 256) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.pool = nn.AvgPool2d(2)

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1 = ResBlock(base, base, time_dim)
        self.down2 = ResBlock(base, base * 2, time_dim)
        self.down3 = ResBlock(base * 2, base * 4, time_dim)
        self.down4 = ResBlock(base * 4, base * 8, time_dim)
        self.mid = ResBlock(base * 8, base * 8, time_dim)

        self.up3 = ResBlock(base * 8 + base * 4, base * 4, time_dim)
        self.up2 = ResBlock(base * 4 + base * 2, base * 2, time_dim)
        self.up1 = ResBlock(base * 2 + base, base, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups(base), base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)

        x1 = self.down1(self.in_conv(x), time_emb)
        x2 = self.down2(self.pool(x1), time_emb)
        x3 = self.down3(self.pool(x2), time_emb)
        x4 = self.down4(self.pool(x3), time_emb)
        h = self.mid(x4, time_emb)

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up3(torch.cat([h, x3], dim=1), time_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2(torch.cat([h, x2], dim=1), time_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1(torch.cat([h, x1], dim=1), time_emb)
        return self.out(h)


class Diffusion:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def _extract(self, values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return values.gather(0, t).view(t.size(0), *((1,) * (x.ndim - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self._extract(self.sqrt_alpha_bars, t, x0) * x0
            + self._extract(self.sqrt_one_minus_alpha_bars, t, x0) * noise
        )

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, int, int, int]) -> torch.Tensor:
        model.eval()
        x = torch.randn(shape, device=self.device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)
            beta_t = self._extract(self.betas, t, x)
            alpha_t = self._extract(self.alphas, t, x)
            alpha_bar_t = self._extract(self.alpha_bars, t, x)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise
            )
            x = mean if step == 0 else mean + torch.sqrt(beta_t) * torch.randn_like(x)

        return x.clamp(-1, 1)


def save_samples(
    model: nn.Module,
    diffusion: Diffusion,
    out_path: Path,
    image_size: int,
    num_samples: int = 4,
) -> None:
    images = diffusion.sample(model, (num_samples, 3, image_size, image_size))
    utils.save_image((images + 1) / 2, out_path, nrow=int(num_samples**0.5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM on labeled STL10.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base", type=int, default=48)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_set = datasets.STL10(
        root=args.data_dir,
        split="train",
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet96(base=args.base).to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x0, _ in train_loader:
            x0 = x0.to(device)
            t = torch.randint(0, args.timesteps, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)
            loss = F.mse_loss(model(xt, t), noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x0.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={avg_loss:.4f}")

        if epoch == 1 or epoch % args.sample_every == 0:
            save_samples(
                model,
                diffusion,
                args.out_dir / f"stl10_sample_epoch_{epoch:03d}.png",
                image_size=args.image_size,
                num_samples=args.sample_count,
            )

    ckpt_path = args.ckpt_dir / "stl10_ddpm.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "base": args.base,
            "timesteps": args.timesteps,
            "image_size": args.image_size,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
