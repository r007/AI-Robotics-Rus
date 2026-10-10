import sys
import os
import argparse

from config import Config
from vlm import create_vlm_engine
from navigator import EmbodiedNavigator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Система навигации воплощённого ИИ: Habitat + YOLO + VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default=os.environ.get("NAV_VLM_MODE", "local"),
        help="Режим VLM: local = локальный Qwen3-VL, api = DashScope API (по умолчанию: local)",
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default=None,
        help="Путь к файлу сцены (.glb); если не задан, используется Config.SCENE_PATH",
    )
    parser.add_argument(
        "targets",
        nargs="?",
        default=None,
        help="Список целевых объектов через запятую, например chair,couch,tv",
    )
    return parser.parse_args()


def print_banner(cfg: Config, mode: str):
    print("=" * 60)
    print(f"  Habitat + YOLO + VLM: навигация воплощённого ИИ")
    print("=" * 60)
    print(f"  Сцена  : {cfg.SCENE_PATH}")
    print(f"  Цели   : {cfg.TARGET_OBJECTS}")
    print(f"  YOLO  : {cfg.YOLO_MODEL}")
    if mode == "local":
        print(f"  VLM    : {cfg.LOCAL_MODEL_NAME}  [локальный вывод]")
    else:
        print(f"  VLM   : {cfg.API_MODEL_NAME}  [API: {cfg.API_BASE_URL}]")
    print(f"  Лимит шагов: {cfg.MAX_STEPS}")
    print("=" * 60)


def main():
    args = parse_args()

    # Собираем конфигурацию
    cfg = Config()
    if args.scene:
        cfg.SCENE_PATH = args.scene
    if args.targets:
        cfg.TARGET_OBJECTS = args.targets.split(",")

    mode = args.mode
    print_banner(cfg, mode)

    # Создаём движок VLM (local или api)
    print(f"\n Инициализация движка VLM [{mode}] ...")
    vlm = create_vlm_engine(cfg, mode=mode)

    # Запускаем навигацию
    EmbodiedNavigator(cfg, vlm).run()


if __name__ == "__main__":
    main()
