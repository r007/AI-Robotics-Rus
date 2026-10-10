from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> str:
    """Сначала пишем H.264; если нет imageio/ffmpeg, откатываемся к OpenCV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("Нет кадров видео для записи.")

    try:
        import imageio.v2 as imageio

        imageio.mimsave(
            path,
            list(frames),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )
        return "imageio/libx264"
    except Exception:
       pass

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Для записи видео нужен imageio+imageio-ffmpeg или opencv-python."
        ) from exc

    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"Кадры видео должны быть изображениями HWC RGB, фактически {first.shape}")
    height, width = first.shape[:2]
    writer = None
    selected_codec = None
    for codec in ("avc1", "H264", "mp4v"):
        candidate = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            selected_codec = codec
            break
        candidate.release()
    if writer is None:
        raise RuntimeError("В OpenCV нет доступного кодировщика MP4.")

    try:
        for frame in frames:
            frame = np.asarray(frame)
            if frame.shape != first.shape:
                raise ValueError(
                    f"Размеры кадров различаются: первый кадр {first.shape}, текущий {frame.shape}"
                )
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return f"opencv/{selected_codec}"
