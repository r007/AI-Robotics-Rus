"""
Generate one 96x96 image from a trained STL10 DDPM checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import utils

from train_stl10_ddpm import Diffusion, UNet96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with a trained STL10 DDPM.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stl10_ddpm.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/generated_stl10.png"))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = int(checkpoint.get("image_size", 96))
    model = UNet96(base=int(checkpoint["base"])).to(device)
    model.load_state_dict(checkpoint["model_state"])

    diffusion = Diffusion(timesteps=int(checkpoint["timesteps"]), device=device)
    with torch.no_grad():
        images = diffusion.sample(model, (args.num_samples, 3, image_size, image_size))
        utils.save_image((images + 1) / 2, args.out, nrow=max(1, int(args.num_samples**0.5)))

    print(f"Saved generated images to {args.out}")


if __name__ == "__main__":
    main()
