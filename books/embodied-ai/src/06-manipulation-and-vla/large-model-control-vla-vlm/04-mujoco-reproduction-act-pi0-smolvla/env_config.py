import os
import re
import platform
import subprocess
from pathlib import Path

def setup_mujoco_path():
    """
    Автоматически решает проблему китайских символов в пути:
    если в текущем пути есть китайские символы, в домашнем каталоге пользователя создаётся английская символическая ссылка и переключается cwd.
    """
    project_root = Path.cwd().resolve()
    # Принудительно задаём английское имя; желательно, чтобы оно соответствовало теме руководства
    link_path = Path.home() / "lerobot-mujoco-tutorial"

    # Регулярное выражение для поиска китайских символов
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', str(project_root)))

    if not has_chinese:
        # Если китайских символов нет и текущий путь не является ссылкой, сразу возвращаем управление
        if "lerobot-mujoco-tutorial" not in str(project_root):
            return project_root

    # Если уже работаем через путь-ссылку, повторять действие не нужно
    if project_root == link_path:
        return project_root

    if not link_path.exists():
        sys_name = platform.system()
        if sys_name == "Windows":
            cmd = ["cmd", "/c", "mklink", "/J", str(link_path), str(project_root)]
            subprocess.run(cmd, capture_output=True)
        elif sys_name in ("Linux", "Darwin"):
            link_path.symlink_to(project_root, target_is_directory=True)
    
    # Переключаем рабочий каталог процесса
    os.chdir(link_path)
    return link_path

# --- Ключевой момент: запускается автоматически при импорте модуля ---
current_path = setup_mujoco_path()
print(f"[Env Info] Среда выполнения готова, текущий рабочий каталог: {current_path}")