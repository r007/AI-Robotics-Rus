#!/usr/bin/env python3
"""Generate mdBook's src/SUMMARY.md from the directory tree of a book.

Every Markdown page becomes an entry titled by its first H1 heading. A directory
that contains README.md is rendered as a page; a directory without it becomes a
draft (title only) so that its children stay nested. Asset folders are skipped.

Usage:
    python3 scripts/make_summary.py books/embodied-ai
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SKIP_DIRS = {
    "assets",
    "images",
    "official_images",
    "official_videos",
    "official_figures",
    "local_images",
    "local_videos",
    "local_reports",
    "logs",
    "__pycache__",
}

# Titles for the top-level chapter directories of the embodied-ai book.
# Keys are directory names (ASCII slugs); values are the Russian chapter titles.
CHAPTER_TITLES = {
    "01-embodied-ai-overview": "Глава 1. Обзор воплощённого ИИ",
    "02-robot-basics-control-and-hand-eye-coordination": "Глава 2. Основы робототехники и координация «рука — глаз»",
    "03-robot-hardware-lerobot-and-rdk-x5": "Глава 3. Аппаратная часть, LeRobot и RDK X5",
    "04-computer-vision-and-3d-reconstruction": "Глава 4. Компьютерное зрение и 3D-реконструкция",
    "05-deep-and-reinforcement-learning": "Глава 5. Глубокое обучение и обучение с подкреплением",
    "06-manipulation-and-vla": "Глава 6. Захват, манипуляции и VLA",
    "07-robot-operation-and-motion-control": "Глава 7. Управление роботом и движением",
    "08-navigation-and-vln": "Глава 8. Навигация и VLN",
    "09-data-and-benchmarks": "Глава 9. Данные и бенчмарки",
    "10-simulation-tools-and-frontiers": "Глава 10. Инструменты симуляции и передовые направления",
    "11-auxiliary-tools": "Глава 11. Вспомогательные инструменты",
    "12-interview-questions": "Глава 12. Вопросы собеседований",
    "13-frontier-project-reproduction": "Глава 13. Воспроизведение передовых проектов",
    "14-references": "Глава 14. Справочные материалы",
    "15-challenges": "Глава 15. Соревнования",
    "16-study-programs": "Глава 16. Учебные программы",
    "17-world-models": "Глава 17. Модели мира",
    "18-drones": "Глава 18. Беспилотники",
    "19-monthly-team-learning": "Глава 19. Ежемесячные групповые занятия",
    "20-wechat-articles": "Глава 20. Материалы для публикации",
    "21-robot-design": "Глава 21. Проектирование роботов и манипуляторов",
    # Hardware Kits (books/hardware-kits)
    "01-start": "Начало работы",
    "02-arduino-nicla": "Arduino Nicla Vision",
    "03-seeed-xiao": "XIAOML Kit (Seeed XIAO ESP32S3)",
    "04-grove-vision": "Grove Vision AI V2",
    "05-raspberry-pi": "Raspberry Pi",
    "06-shared": "Общие лабораторные работы",
}

# Titles for folders that have no README.md of their own, keyed by folder name.
DIRECTORY_TITLES = {
    "code": "Код",
    "docs": "Документы",
    "results": "Результаты",
    "resources": "Ресурсы",
    "appendices": "Приложения",
    "images": "Изображения",
    "getting_started": "Начало работы",
    "supplement-cartpole-control-tutorial": "Дополнение: управление Cartpole",
    "supplement-01joycon-control": "Дополнение 01. Управление через Joy-Con",
    "large-model-control-vla-vlm": "Управление с помощью больших моделей: VLA и VLM",
    "01SmolVLA-LIBERO": "SmolVLA и LIBERO",
    "02-openvla-reproduction": "Воспроизведение OpenVLA",
    "03-rt-series-paper-interpretation-and-code-analysis": "Серия RT: разбор статей и кода",
    "05DiT4DiT-LIBERO": "DiT4DiT и LIBERO",
    "Locomotion": "Локомоция",
    "01-navigation-algorithms-practical-guide": "Алгоритмы навигации: подробный разбор и практика",
    "02-basics-of-simulation-environment": "Основы среды моделирования",
    "habitat-navigation-environment": "Среда навигации Habitat",
    "03-frontier-vln-reproduction": "Воспроизведение передовых проектов VLN",
    "01VLNCE": "VLN-CE",
    "01-isaac-deployment-and-gr00t-practices": "Развёртывание Isaac и практика с GR00T",
    "01-isaac-sim5-0-installation": "Установка Isaac Sim 5.0",
    "09-sim1-soft-body-simulation-and-data-generation": "SIM1: моделирование мягких тел и генерация данных",
    "10-internverse-tutorial": "Руководство по InternVerse",
    "MetaSim-roboverse": "MetaSim и RoboVerse",
    "RoboGen": "RoboGen",
    "challenge-competition": "Соревнование",
    "cosmos": "Cosmos",
    "genesis-simulation-environment-configuration": "Настройка среды моделирования Genesis",
    "maniskill-detailed-documentation": "ManiSkill: подробная документация",
    "instance-document": "Документы с примерами",
    "docker-related": "Docker",
    "git-related-usage": "Git",
    "lerobot-related": "LeRobot",
    "openpi-related": "OpenPI",
    "package-management-tools-such-as-conda-mamba-pip-uv": "Управление пакетами: conda, mamba, pip, uv",
    "vscode-usage-related": "VS Code",
    "wsa": "WSA",
    "wsl": "WSL",
    "drone-large-model-groundingdino-practice": "Беспилотник и большая модель: практика с GroundingDINO",
    "embodied-navigation-practical-introduction": "Навигация воплощённого ИИ: практическое введение",
    "act_seed1_best_submission": "Лучшая отправка ACT (seed 1)",
    "AgiBot_World_Model": "AgiBot World Model",
    "rocm": "ROCm",
    "01-task-information": "Материалы задания",
    "02-executable-code": "Исполняемый код",
    "01-diffusion-mathematics-and-problem-analysis": "Математика диффузии и постановка задачи",
    "02-vae-explanation-and-code": "VAE: объяснение и код",
    "03-ddpm-explanation-and-code": "DDPM: объяснение и код",
    "leworldmodel-analysis-and-reproduction": "LeWorldModel: разбор и воспроизведение",
}

# Pages of the book that describe the translation itself; they go to the last part.
ABOUT_PAGES = (
    ("notice.md", "Авторство и лицензия"),
    ("provenance.md", "Происхождение материалов"),
)

_NATURAL = re.compile(r"(\d+)")


def natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in _NATURAL.split(name)]


def page_title(path: Path) -> str:
    """Return the first H1 heading of a Markdown file, or the file stem."""
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("# "):
                    title = line[2:].strip().replace("**", "")
                    return re.sub(r"[\[\]|`]", "", title).strip()
    except (OSError, UnicodeDecodeError):
        pass
    return path.stem


def has_markdown(directory: Path) -> bool:
    return any(
        path.suffix == ".md" and path.name != "SUMMARY.md"
        for path in directory.rglob("*")
        if path.is_file()
    )


def entry_title(name: str) -> str:
    return name.replace("**", "").replace("[", "").replace("]", "")


def render_dir(src: Path, directory: Path, depth: int, lines: list[str]) -> None:
    children = sorted(
        (child for child in directory.iterdir() if not child.name.startswith(".")),
        key=lambda child: (child.name != "README.md", natural_key(child.name)),
    )
    indent = "  " * depth
    for child in children:
        rel = child.relative_to(src).as_posix()
        # A nested directory's README is already the directory's own entry.
        if depth > 0 and child.name == "README.md":
            continue
        if child.is_file() and child.suffix == ".md" and child.name != "SUMMARY.md":
            title = page_title(child)
            lines.append(f"{indent}- [{entry_title(title)}]({rel})")
        elif child.is_dir() and child.name not in SKIP_DIRS and has_markdown(child):
            readme = child / "README.md"
            if readme.exists():
                title = page_title(readme)
            else:
                title = DIRECTORY_TITLES.get(child.name, child.name)
            if readme.exists():
                lines.append(f"{indent}- [{entry_title(title)}]({rel}/README.md)")
            else:
                lines.append(f"{indent}- [{entry_title(title)}]()")
            render_dir(src, child, depth + 1, lines)


def build(src: Path) -> str:
    lines = ["# Содержание", ""]
    intro = src / "README.md"
    if intro.exists():
        lines.append(f"[Введение]({intro.relative_to(src).as_posix()})")
        lines.append("")

    top = sorted(
        (child for child in src.iterdir() if child.is_dir() and child.name not in SKIP_DIRS),
        key=lambda child: natural_key(child.name),
    )
    for chapter in top:
        title = CHAPTER_TITLES.get(chapter.name, chapter.name)
        lines.append(f"# {title}")
        lines.append("")
        chapter_lines: list[str] = []
        render_dir(src, chapter, 0, chapter_lines)
        lines.extend(chapter_lines)
        lines.append("")

    loose = sorted(
        (
            child
            for child in src.iterdir()
            if child.is_file()
            and child.suffix == ".md"
            and child.name not in {"README.md", "SUMMARY.md", *(name for name, _ in ABOUT_PAGES)}
        ),
        key=lambda child: natural_key(child.name),
    )
    if loose:
        lines.append("# Приложения")
        lines.append("")
        for child in loose:
            lines.append(f"- [{entry_title(page_title(child))}]({child.name})")
        lines.append("")

    about = [(name, title) for name, title in ABOUT_PAGES if (src / name).is_file()]
    if about:
        lines.append("# О переводе")
        lines.append("")
        lines.extend(f"- [{title}]({name})" for name, title in about)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    book = Path(argv[1]).resolve()
    src = book / "src"
    if not src.is_dir():
        print(f"error: {src} does not exist", file=sys.stderr)
        return 1
    summary = build(src)
    (src / "SUMMARY.md").write_text(summary, encoding="utf-8")
    pages = summary.count("](")
    print(f"wrote {src / 'SUMMARY.md'} ({pages} links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
