"""Вводный урок, Demo 2: генерация видео движения суставов манипулятора в upstream-сцене MuJoCo.

Скрипт зависит от external/mujoco_pnp/asset/example_scene_y2.xml и позволяет проверить,
что upstream-проект, ассеты сцены и официальный Renderer уже подготовлены.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys


def parse_args() -> argparse.Namespace:
    topic_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Render the tutorial robot-arm motion scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=topic_root / "external" / "mujoco_pnp",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=topic_root / "outputs" / "demo_videos" / "arm_swing.mp4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "This demo requires the mujoco and numpy Python packages."
        ) from exc

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from video_io import write_video

    xml_path = args.project_root.expanduser() / "asset" / "example_scene_y2.xml"
    if not xml_path.exists():
        raise FileNotFoundError(
            f"Файл сцены не найден: {xml_path}. Сначала подготовьте upstream-проект по external/README.md."
        )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = 1 / 60.0

    print("Сцена манипулятора успешно скомпилирована")
    print(f"Суставов={model.nq}, степеней свободы={model.nv}, body={model.nbody}, geom={model.ngeom}")
    print(f"Камеры={[model.camera(i).name for i in range(model.ncam)]}")
    print(f"Актуаторы={[model.actuator(i).name for i in range(model.nu)]}")

    joint_names = [model.joint(i).name for i in range(model.njnt)]
    if "joint1" not in joint_names:
        raise ValueError(f"В сцене не найден joint1; текущие суставы: {joint_names}")
    arm_joint_id = joint_names.index("joint1")
    arm_qpos_address = model.jnt_qposadr[arm_joint_id]

    dof = 6
    amplitude = np.array([0.6, 0.6, 0.8, 0.5, 0.5, 0.3])
    frequency = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    phase = np.array([0.0, 1.0, 0.5, 1.5, 2.0, 0.8])
    kp = 20.0
    renderer = mujoco.Renderer(model, height=480, width=640)

    frames = []
    for step in range(240):
        t = step * model.opt.timestep
        target = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        qpos = data.qpos[arm_qpos_address : arm_qpos_address + dof]
        data.ctrl[:dof] = kp * (target - qpos)
        mujoco.mj_step(model, data)
        if step % 4 == 0:
            renderer.update_scene(data, camera="agentview")
            frames.append(renderer.render().copy())

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    codec = write_video(output, frames, fps=15)
    nonblack = sum(np.mean(frame) > 5 for frame in frames)
    print(
        f"Видео сохранено: {output}; codec={codec}; {len(frames)} кадров, "
        f"ненулевых кадров {nonblack}/{len(frames)}"
    )
    print(
        "Если запускаете в удалённом Code Server, скачайте видео из outputs/demo_videos/ на локальную машину для воспроизведения."
    )


if __name__ == "__main__":
    main()
