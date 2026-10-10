from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def md(value: str):
    return nbf.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbf.v4.new_code_cell(value.strip())


SETUP = r'''
from pathlib import Path
import json
import os
import platform
import shlex
import socket
import subprocess


def locate_topic_root():
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / "02-可运行代码").is_dir() and (base / "01-任务资料").is_dir():  # каталоги upstream, не переводить: 02-可运行代码 — исполняемый код, 01-任务资料 — материалы задач
            return base
        group = base / "16-专题组队学习"  # каталог группового обучения upstream, не переводить
        if group.is_dir():
            matches = list(group.glob("*/02-可运行代码"))  # см. комментарий выше
            if matches:
                return matches[0].parent
    configured = os.getenv("MICRODUCK_TOPIC_ROOT")
    if configured and Path(configured).is_dir():
        return Path(configured).resolve()
    raise RuntimeError("Не найден каталог темы; запустите Jupyter из 03-Notebook или задайте MICRODUCK_TOPIC_ROOT.")


TOPIC_ROOT = locate_topic_root()
PLAYGROUND_ROOT = TOPIC_ROOT / "02-可运行代码" / "microduck-playground-stilts"  # путь upstream, не переводить
OUTPUT_ROOT = TOPIC_ROOT / "03-Notebook" / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("Каталог темы:", TOPIC_ROOT)
print("Исполняемый код:", PLAYGROUND_ROOT)
print("Выходной каталог:", OUTPUT_ROOT)
print("Станция Jupyter:", socket.gethostname(), platform.system(), platform.machine())
print("Адрес RDK:", os.getenv("RDK_HOST", "192.168.8.128"), "пользователь: sunrise")
'''


def task_cells(model_hint: str | None, task_id: str | None, sim_command: str, direct_connect: bool = False, video_hint: str | None = None, keyframe_hint: str | None = None, bpu_task: str | None = None):
    hint = repr(model_hint) if model_hint else "None"
    task = repr(task_id) if task_id else "None"
    is_fastsac = bpu_task == "ball_balance"
    obs_dim = 54 if is_fastsac else 61
    if direct_connect:
        stage3_md = md("""
## 3. Политика навигации: прямое подключение готовой модели

Шаг навигации и развёртывания не переобучает модель, а сразу использует готовый ONNX. По умолчанию читается `outputs/microduck_demo_latest.onnx` в каталоге этого Notebook; через `MICRODUCK_NAV_ONNX` можно указать другую уже обученную навигационную политику.
""")
        stage3_code = code(r'''
NAV_ONNX = os.getenv("MICRODUCK_NAV_ONNX", "")
default_nav_onnx = OUTPUT_ROOT / "microduck_demo_latest.onnx"
if NAV_ONNX:
    MODEL_PATH = Path(NAV_ONNX).expanduser().resolve()
elif default_nav_onnx.exists():
    MODEL_PATH = default_nav_onnx
print("Политика навигации подключена:", MODEL_PATH if MODEL_PATH and MODEL_PATH.exists() else "не найдена")
if MODEL_PATH is None or not MODEL_PATH.exists():
    print("Задайте MICRODUCK_NAV_ONNX=/path/to/navigation.onnx и запустите эту ячейку снова.")
''')
    elif is_fastsac:
        stage3_md = md("""
## 3. Smoke-обучение FastSAC и экспорт ONNX

Балансировка на шаре использует FastSAC из MotrixLab, а не PPO/LSTM. По умолчанию в `.venv` MotrixLab на Ubuntu запускаются 5 smoke-итераций с 64 параллельными средами, а текущий run экспортируется в `outputs/motrix_ball_balance_latest.onnx`. Если после переобучения нужно новое видео на плате, сначала заново выполните ячейку компиляции X5 HBM; старый HBM использовать нельзя.
""")
        stage3_code = code(r'''
import shutil
import sys

MOTRIX_ROOT = Path(os.getenv("MOTRIX_ROOT", "/home/ubuntu/workspaces/MotrixLab")).expanduser()
MOTRIX_PYTHON = Path(os.getenv("MOTRIX_PYTHON", str(MOTRIX_ROOT / ".venv/bin/python"))).expanduser()
MOTRIX_RUNNER = PLAYGROUND_ROOT / "scripts" / "motrix_runner.py"
FASTSAC_ENVS = int(os.getenv("MICRODUCK_FASTSAC_ENVS", "64"))
FASTSAC_ITERATIONS = int(os.getenv("MICRODUCK_FASTSAC_ITERATIONS", "5"))
RUN_FASTSAC_SMOKE = os.getenv("MICRODUCK_RUN_FASTSAC_SMOKE", "1") == "1"
FASTSAC_ONNX = OUTPUT_ROOT / "motrix_ball_balance_latest.onnx"
print("MotrixLab:", MOTRIX_ROOT)
print("FastSAC Python:", MOTRIX_PYTHON)
print("Фиксированный выход:", FASTSAC_ONNX)

if not RUN_FASTSAC_SMOKE:
    print("Smoke-обучение FastSAC пропущено: чтобы запустить, задайте MICRODUCK_RUN_FASTSAC_SMOKE=1 и выполните ячейку снова.")
elif not MOTRIX_ROOT.is_dir() or not MOTRIX_PYTHON.is_file():
    print("Не найдены MotrixLab или его окружение Python; сначала установите MotrixLab на Ubuntu.")
else:
    train_cmd = [
        str(MOTRIX_PYTHON), str(MOTRIX_RUNNER), "train", str(MOTRIX_ROOT / "scripts/train.py"),
        "task=microduck-ball-balance/motrix.fastsac",
        f"num_envs={FASTSAC_ENVS}",
        "play=false", "render=false", "algo.asynchronous=false",
        "algo.device=cuda", "algo.agent.compile=false", "algo.agent.amp=false",
        # Keep the default run genuinely smoke-sized. The replay buffer must
        # be populated before FastSAC can take its first update.
        "algo.agent.learning_starts=0", "algo.agent.buffer_size=8", "algo.agent.batch_size=64",
        "algo.agent.num_updates=1", f"algo.trainer.num_learning_iterations={FASTSAC_ITERATIONS}",
    ]
    print("Запуск smoke-обучения FastSAC:", " ".join(shlex.quote(x) for x in train_cmd))
    train_result = subprocess.run(train_cmd, cwd=MOTRIX_ROOT, text=True)
    if train_result.returncode != 0:
        raise RuntimeError(f"Smoke-обучение FastSAC завершилось с ошибкой, returncode={train_result.returncode}")
    run_root = MOTRIX_ROOT / "runs" / "microduck-ball-balance" / "motrix" / "torch" / "fastsac"
    run_dirs = sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise FileNotFoundError(f"Обучение прошло успешно, но run FastSAC не найден: {run_root}")
    run_dir = run_dirs[-1]
    motrix_export = Path("outputs") / "motrix_ball_balance_latest.onnx"
    export_cmd = [
        str(MOTRIX_PYTHON), str(MOTRIX_RUNNER), "export", str(MOTRIX_ROOT / "scripts/export_onnx.py"),
        f"run_dir={run_dir}", f"output={motrix_export.as_posix()}", "opset=11",
    ]
    print("Экспорт ONNX FastSAC:", " ".join(shlex.quote(x) for x in export_cmd))
    export_result = subprocess.run(export_cmd, cwd=MOTRIX_ROOT, text=True)
    exported_onnx = MOTRIX_ROOT / motrix_export
    if export_result.returncode != 0 or not exported_onnx.is_file():
        raise RuntimeError("Экспорт ONNX FastSAC не удался.")
    shutil.copy2(exported_onnx, FASTSAC_ONNX)
    MODEL_PATH = FASTSAC_ONNX
    print("PASS-fastsac-model:", MODEL_PATH)
    print("Фактический run обучения:", run_dir)
''')
    else:
        stage3_md = md("""
## 3. Фиксированная демо-модель: восстановить и smoke-обучить 10 итераций

Чтобы каждый запуск на стриме давал один и тот же воспроизводимый результат, эта ячейка по умолчанию восстанавливает проверенный checkpoint ходьбы и продолжает обучать 10 итераций PPO на GPU с 64 параллельными средами, затем перезаписывает `outputs/microduck_demo_latest.pt` и `outputs/microduck_demo_latest.onnx`. Исходный `model_5999.pt` не изменяется.

10 итераций — это не «переобучение новой способности», а проверка всей цепочки `checkpoint -> MuJoCo/Warp -> обновление PPO -> checkpoint -> ONNX`. По умолчанию каждая итерация собирает `64 × 24 = 1536` шагов среды, поэтому всего обновляется около 15360 переходов.
""")
        stage3_code = code(r'''
import shutil
import sys

TRAIN_ROOT = Path(os.getenv("MICRODUCK_WORKSPACE", "/home/ubuntu/workspaces/microduck_rl"))
TRAIN_TASK = os.getenv("MICRODUCK_TRAIN_TASK", "Mjlab-Velocity-Flat-MicroDuck")
TRAIN_SOURCE_RUN = os.getenv(
    "MICRODUCK_TRAIN_RUN",
    "2026-09-03_20-05-56_every-embodied-4096x6000",
)
TRAIN_SOURCE_CHECKPOINT = os.getenv("MICRODUCK_TRAIN_CHECKPOINT", "model_5999.pt")
TRAIN_ENVS = int(os.getenv("MICRODUCK_TRAIN_ENVS", "64"))
TRAIN_ITERATIONS = int(os.getenv("MICRODUCK_TRAIN_ITERATIONS", "10"))
RUN_TRAIN_SMOKE = os.getenv("MICRODUCK_RUN_TRAIN_SMOKE", "1") == "1"
DEMO_PT = OUTPUT_ROOT / "microduck_demo_latest.pt"
DEMO_ONNX = OUTPUT_ROOT / "microduck_demo_latest.onnx"

source_path = TRAIN_ROOT / "logs" / "rsl_rl" / "velocity" / TRAIN_SOURCE_RUN / TRAIN_SOURCE_CHECKPOINT
print("Рабочая область обучения:", TRAIN_ROOT)
print("Восстанавливаемый checkpoint:", source_path)
print("Фиксированный выход:", DEMO_PT, DEMO_ONNX)

if not RUN_TRAIN_SMOKE:
    print("Smoke-обучение пропущено: чтобы запустить, задайте MICRODUCK_RUN_TRAIN_SMOKE=1 и выполните эту ячейку снова.")
elif not TRAIN_ROOT.is_dir():
    print("Не найдена рабочая область обучения Ubuntu; запустите в Jupyter на Ubuntu или задайте MICRODUCK_WORKSPACE.")
elif not source_path.is_file():
    print("Не найден checkpoint для восстановления; задайте MICRODUCK_TRAIN_RUN / MICRODUCK_TRAIN_CHECKPOINT.")
else:
    train_cmd = [
        sys.executable, "-m", "mjlab.scripts.train", TRAIN_TASK,
        "--env.scene.num-envs", str(TRAIN_ENVS),
        "--agent.max-iterations", str(TRAIN_ITERATIONS),
        "--agent.resume", "True",
        "--agent.load-run", TRAIN_SOURCE_RUN,
        "--agent.load-checkpoint", TRAIN_SOURCE_CHECKPOINT,
        "--agent.experiment-name", "velocity",
        "--agent.run-name", "demo-latest-10-step",
        "--agent.save-interval", str(TRAIN_ITERATIONS),
        "--video", "False",
    ]
    train_env = os.environ.copy()
    train_env["WANDB_MODE"] = "offline"
    print("Запуск smoke-обучения на GPU:", " ".join(shlex.quote(x) for x in train_cmd))
    train_result = subprocess.run(train_cmd, cwd=TRAIN_ROOT, env=train_env, text=True)
    if train_result.returncode != 0:
        print("Обучение завершилось с ошибкой, фиксированная демо-модель не перезаписана. returncode:", train_result.returncode)
    else:
        demo_runs = sorted(
            (TRAIN_ROOT / "logs" / "rsl_rl" / "velocity").glob("*_demo-latest-10-step"),
            key=lambda p: p.stat().st_mtime,
        )
        if not demo_runs:
            raise FileNotFoundError("Обучение прошло успешно, но не найден выходной каталог demo-latest-10-step.")
        demo_run = demo_runs[-1]
        checkpoints = sorted(
            demo_run.glob("model_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"В выходном каталоге обучения нет checkpoint: {demo_run}")
        latest_checkpoint = checkpoints[-1]
        shutil.copy2(latest_checkpoint, DEMO_PT)
        export_cmd = [
            sys.executable, "scripts/export.py", TRAIN_TASK,
            "--checkpoint-file", str(latest_checkpoint),
            "--onnx-file", str(DEMO_ONNX),
        ]
        print("Экспорт фиксированного ONNX:", " ".join(shlex.quote(x) for x in export_cmd))
        export_result = subprocess.run(export_cmd, cwd=TRAIN_ROOT, env=train_env, text=True)
        if export_result.returncode != 0:
            print("Ошибка экспорта ONNX, checkpoint сохранён, returncode:", export_result.returncode)
        else:
            MODEL_PATH = DEMO_ONNX
            print("PASS-demo-model:", MODEL_PATH)
            print("Фактически использован checkpoint:", latest_checkpoint)
''')
    if is_fastsac:
        hbm_stage = [
            md("""
## 3b. Компиляция X5 HBM

Обученный ONNX нельзя напрямую передать RDK X5. Ниже фиксируем batch входа и выхода равным 1 и запускаем официальный CPU Docker-инструментарий X5 на Ubuntu: `hb_mapper makertbin --model-type onnx`. Сгенерированный `ball_balance.opset11.bin` перезапишет каталог компиляции; видео следующего раздела использует именно этот новый HBM.
"""),
            code(r'''
import shutil

BPU_MODEL_ROOT = Path(os.getenv("MICRODUCK_BPU_MODEL_ROOT", "/home/ubuntu/workspaces/microduck_bpu_models")).expanduser()
BPU_CONVERTED = BPU_MODEL_ROOT / "converted"
BPU_COMPILED = BPU_MODEL_ROOT / "compiled" / "ball_balance"
BPU_STATIC_ONNX = BPU_CONVERTED / "motrix_ball_balance.static.opset11.onnx"
BPU_CONFIG = BPU_COMPILED / ".fast_perf" / "ball_balance.opset11_config.yaml"
BPU_OUTPUT = BPU_COMPILED / "model_output" / "ball_balance.opset11.bin"
BPU_DOCKER_IMAGE = os.getenv("MICRODUCK_BPU_DOCKER_IMAGE", "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8")

def workspace_container_path(path):
    workspace = Path("/home/ubuntu/workspaces")
    try:
        return Path("/work") / path.resolve().relative_to(workspace)
    except ValueError as exc:
        raise RuntimeError(f"HBM-файл должен находиться в {workspace}: {path}") from exc

if not FASTSAC_ONNX.is_file():
    print("Компиляция HBM пропущена: сначала успешно выполните обучение и экспорт FastSAC в разделе 3.")
elif shutil.which("docker") is None:
    print("Компиляция HBM пропущена: на Ubuntu не найден docker.")
else:
    import onnx

    BPU_CONVERTED.mkdir(parents=True, exist_ok=True)
    BPU_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    BPU_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for stale_file in BPU_OUTPUT.parent.iterdir():
        if stale_file.is_file():
            stale_file.unlink()
    static_model = onnx.load(str(FASTSAC_ONNX))
    for value in list(static_model.graph.input) + list(static_model.graph.output):
        value.type.tensor_type.shape.dim[0].dim_value = 1
        value.type.tensor_type.shape.dim[0].ClearField("dim_param")
    onnx.checker.check_model(static_model)
    onnx.save(static_model, str(BPU_STATIC_ONNX))
    BPU_CONFIG.write_text(
        """calibration_parameters:
  cal_data_type: ''
  calibration_type: skip
  optimization: run_fast
  per_channel: false
  preprocess_on: false
  run_on_bpu: ''
  run_on_cpu: ''
compiler_parameters:
  compile_mode: latency
  core_num: 1
  debug: true
  jobs: 0
  max_time_per_fc: 0
  optimize_level: O3
input_parameters:
  input_layout_rt: NCHW
  input_layout_train: NCHW
  input_name: obs
  input_shape: 1x54
  input_space_and_range: ''
  input_type_rt: featuremap
  input_type_train: featuremap
  norm_type: no_preprocess
model_parameters:
  layer_out_dump: false
  march: bayes-e
  onnx_model: /work/microduck_bpu_models/converted/motrix_ball_balance.static.opset11.onnx
  output_model_file_prefix: ball_balance.opset11
  remove_node_type: Quantize;Transpose;Dequantize;Cast;Reshape;Softmax;DequantizeFilter
  set_node_data_type: {}
  working_dir: /work/microduck_bpu_models/compiled/ball_balance/model_output
""",
        encoding="utf-8",
    )
    docker_cmd = [
        "docker", "run", "--rm", "-v", "/home/ubuntu/workspaces:/work",
        BPU_DOCKER_IMAGE, "sh", "-lc",
        "hb_mapper makertbin --config "
        + shlex.quote(str(workspace_container_path(BPU_CONFIG)))
        + " --model-type onnx",
    ]
    print("Запуск компиляции X5 HBM:", " ".join(shlex.quote(x) for x in docker_cmd))
    compile_result = subprocess.run(docker_cmd, text=True)
    if compile_result.returncode != 0 or not BPU_OUTPUT.is_file():
        raise RuntimeError("Компиляция X5 HBM не удалась.")
    os.environ["MICRODUCK_BPU_HBM"] = str(BPU_OUTPUT)
    print("PASS-X5-HBM:", BPU_OUTPUT)
'''),
        ]
    else:
        hbm_stage = []

    if is_fastsac:
        export_template = "python scripts/export_onnx.py run_dir=<FASTSAC_RUN_DIR> output=outputs/motrix_ball_balance_latest.onnx opset=11"
    else:
        export_template = (
            "uv run python scripts/export.py "
            + (task_id or "<TASK_ID>")
            + "\\\n  --checkpoint-file <CHECKPOINT.pt>\\\n  --onnx-file outputs/policy.onnx\\\n  --num-envs 1 --video --video-length 250"
        )

    return [
        md(f"""
## 1. Контракт модели и локальная проверка

Целевой контракт: actor `obs[1, {obs_dim}] -> action[1, 14]`. Если для этой задачи в тему не загружены checkpoint или ONNX, Notebook явно покажет «модель не предоставлена» и не будет выдавать модель другой задачи за результат текущей.
"""),
        code(f'''
TASK_ID = {task}
MODEL_HINT = {hint}
MODEL_PATH = Path(os.getenv("MICRODUCK_ONNX", "")) if os.getenv("MICRODUCK_ONNX") else None
if MODEL_PATH is None and MODEL_HINT:
    candidate = TOPIC_ROOT / MODEL_HINT
    if candidate.exists():
        MODEL_PATH = candidate
if MODEL_PATH is None and not MODEL_HINT and not TASK_ID:
    candidates = sorted(TOPIC_ROOT.glob("01-任务资料/**/*.onnx"))  # путь upstream, не переводить
    MODEL_PATH = candidates[0] if candidates else None
print("Задача:", TASK_ID or "интерфейс/веб-демо")
print("ONNX:", MODEL_PATH if MODEL_PATH else "не предоставлен")
if TASK_ID and MODEL_PATH is None:
    print("Для этой задачи в тему пока не загружен специальный ONNX; задайте MICRODUCK_ONNX и снова запустите проверку модели.")
'''),
        code(r'''
def inspect_onnx(path):
    try:
        import onnx
    except ImportError:
        print("Нет onnx: запустите Jupyter в окружении microduck-playground.")
        return None
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    def shape(value):
        return [d.dim_value if d.dim_value else (d.dim_param or "?") for d in value.type.tensor_type.shape.dim]
    result = {
        "inputs": [(item.name, shape(item)) for item in model.graph.input],
        "outputs": [(item.name, shape(item)) for item in model.graph.output],
        "nodes": len(model.graph.node),
        "metadata": {item.key: item.value for item in model.metadata_props},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

contract = inspect_onnx(MODEL_PATH) if MODEL_PATH else None
'''),
        md(f"""
## 2. Точка входа экспорта ONNX

Обучение даёт checkpoint политики; при экспорте нужно зафиксировать в ONNX и actor, и нормализацию наблюдений. Ниже — шаблон команды для этой задачи. `--video` записывает симуляционный повтор на локальной машине; полный видеофайл сохраняется в `outputs/` и не отправляется в Git.

```bash
{export_template}
```
"""),
        code(r'''
EXPORT_COMMAND = "python scripts/export_onnx.py run_dir=<FASTSAC_RUN_DIR> output=outputs/motrix_ball_balance_latest.onnx opset=11" if TASK_ID == "microduck-ball-balance" else ("uv run python scripts/export.py " + (TASK_ID or "<TASK_ID>") + " --checkpoint-file <CHECKPOINT.pt> --onnx-file outputs/policy.onnx --num-envs 1 --video --video-length 250")
print(EXPORT_COMMAND)
print("Проверка до и после экспорта: размерность наблюдений, порядок действий, нормализация и clip действий.")
'''),
        stage3_md,
        stage3_code,
        *hbm_stage,
        md("""
## 4. Локальный базовый инференс ONNX

Этот шаг только проверяет, что граф ONNX работает на этой машине, а выходы имеют правильную форму и конечные значения; это не оценка качества обучения и не приёмка BPU.
"""),
        code(r'''
def onnx_local_smoke(path):
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print("Нет локальных зависимостей для инференса:", exc)
        return None
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {}
    for item in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in item.shape]
        feeds[item.name] = np.zeros(shape, dtype=np.float32)
    outputs = session.run(None, feeds)
    info = {"providers": session.get_providers(), "inputs": [(x.name, x.shape) for x in session.get_inputs()], "outputs": [(x.name, x.shape) for x in session.get_outputs()], "finite": all(np.isfinite(x).all() for x in outputs)}
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    return outputs

if MODEL_PATH:
    local_outputs = onnx_local_smoke(MODEL_PATH)
else:
    print("Пропуск: ONNX для этой задачи ещё не предоставлен.")
'''),
        md("""
## 5. Видео BPU-in-the-loop на RDK X5

Эта ячейка показывает только новый замкнутый MP4 с платы и не читает эталонные GIF или ключевые кадры. Ubuntu выполняет физику MuJoCo и рендеринг EGL; на каждом управляющем шаге наблюдение задачи отправляется на RDK, RDK вычисляет 14-мерное действие на BPU с HBM текущей задачи и возвращает его на Ubuntu. Поэтому видео — настоящий замкнутый цикл: действия вычисляет BPU на плате, а картинку симуляции — Ubuntu.

Перед запуском автоматически перезапускается сервис политики задач на RDK. При переключении Notebook эту ячейку нужно выполнять заново, чтобы HBM на плате соответствовал текущей задаче.
"""),
code(f'''
import hashlib
import sys

BPU_TASK = {bpu_task!r}
BPU_VIDEO_STEPS = int(os.getenv("MICRODUCK_BPU_VIDEO_STEPS", "250"))
BPU_VIDEO_WIDTH = int(os.getenv("MICRODUCK_BPU_VIDEO_WIDTH", "640"))
BPU_VIDEO_HEIGHT = int(os.getenv("MICRODUCK_BPU_VIDEO_HEIGHT", "480"))
BPU_PORT = int(os.getenv("MICRODUCK_BPU_PORT", "8765"))
RDK_CONTROL_PORT = int(os.getenv("MICRODUCK_BPU_CONTROL_PORT", "8766"))
RDK_HOST = os.getenv("RDK_HOST", "192.168.8.128")
BPU_HBM_DEFAULTS = {{
    "ball_balance": "/home/ubuntu/workspaces/microduck_bpu_models/compiled/ball_balance/model_output/ball_balance.opset11.bin",
    "basketball": "/home/ubuntu/workspaces/microduck_bpu_models/compiled/basketball/model_output/basketball.opset11.bin",
    "walking": "/home/ubuntu/workspaces/microduck_bpu_models/compiled/walking/model_output/walking.opset11.bin",
    "perturbation": "/home/ubuntu/workspaces/microduck_bpu_models/compiled/walking/model_output/walking.opset11.bin",
    "stilt": "/home/ubuntu/workspaces/microduck_bpu_models/compiled/stilts25/model_output/stilts25.opset11.bin",
    "swing": "/home/ubuntu/workspaces/microduck_bpu_models/compiled/swing/model_output/swing.opset11.bin",
    "ladder": "/home/ubuntu/workspaces/microduck_bpu_models/compiled/ladder/model_output/ladder.opset11.bin",
}}
RDK_HBM_DEFAULTS = {{
    "ball_balance": "/home/sunrise/microduck_policy/hbm/ball_balance.bin",
    "basketball": "/home/sunrise/microduck_policy/hbm/basketball.bin",
    "walking": "/home/sunrise/microduck_policy/hbm/walking.bin",
    "perturbation": "/home/sunrise/microduck_policy/hbm/walking.bin",
    "stilt": "/home/sunrise/microduck_policy/hbm/stilts25.bin",
    "swing": "/home/sunrise/microduck_policy/hbm/swing.bin",
    "ladder": "/home/sunrise/microduck_policy/hbm/ladder.bin",
}}
BPU_HBM = Path(os.getenv("MICRODUCK_BPU_HBM", BPU_HBM_DEFAULTS.get(BPU_TASK, ""))) if BPU_TASK else None
RDK_HBM = RDK_HBM_DEFAULTS.get(BPU_TASK, "") if BPU_TASK else ""
RDK_SERVER_SCRIPT = os.getenv("RDK_BPU_SERVER_SCRIPT", "/home/sunrise/microduck_policy/rdk_bpu_policy_server.py")
MOTRIX_ROOT = Path(os.getenv("MOTRIX_ROOT", "/home/ubuntu/workspaces/MotrixLab")).expanduser() if BPU_TASK == "ball_balance" else None
MOTRIX_PYTHON = Path(os.getenv("MOTRIX_PYTHON", str(MOTRIX_ROOT / ".venv/bin/python"))).expanduser() if MOTRIX_ROOT else None
MOTRIX_RUNNER = PLAYGROUND_ROOT / "scripts" / "motrix_runner.py" if BPU_TASK == "ball_balance" else None
BPU_VIDEO_SCRIPT = PLAYGROUND_ROOT / "scripts" / ("motrix_bpu_inloop_video.py" if BPU_TASK == "ball_balance" else "bpu_inloop_video.py")
BPU_VIDEO_OUTPUT = OUTPUT_ROOT / "bpu_videos" / f"{{BPU_TASK}}_bpu_latest.mp4" if BPU_TASK else None

def restart_rdk_policy_server():
    if not BPU_TASK or not RDK_HBM:
        return False
    try:
        if BPU_HBM is not None and BPU_HBM.is_file():
            payload = BPU_HBM.read_bytes()
            request = {{
                "op": "upload_and_switch",
                "task": BPU_TASK,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }}
        else:
            payload = b""
            request = {{"op": "switch", "task": BPU_TASK}}
        with socket.create_connection((RDK_HOST, RDK_CONTROL_PORT), timeout=8) as control:
            control.sendall((json.dumps(request) + "\\n").encode("utf-8"))
            if payload:
                for offset in range(0, len(payload), 1024 * 1024):
                    control.sendall(payload[offset:offset + 1024 * 1024])
            response = json.loads(control.makefile("rb").readline().decode("utf-8"))
        if response.get("ok"):
            action = "загружено и переключено" if payload else "переключено"
            print("Сервис политики задач RDK " + action + ":", BPU_TASK, "->", response.get("model"), "pid=", response.get("pid"))
            return True
        print("Сервис BPU-управления RDK отклонил переключение:", response)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("Сервис BPU-управления RDK недоступен, пробуем SSH:", exc)
    remote = "sunrise@" + RDK_HOST
    command = (
        "pkill -f '^python3 .*rdk_bpu_policy_server.py' >/dev/null 2>&1 || true; "
        "sleep 1; "
        "nohup python3 " + shlex.quote(RDK_SERVER_SCRIPT) +
        " --model " + shlex.quote(RDK_HBM) +
        " --port " + str(BPU_PORT) +
        " >/tmp/microduck_bpu_server.log 2>&1 </dev/null &"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", remote, "bash -lc " + shlex.quote(command)],
        capture_output=True, text=True,
    )
    if result.returncode:
        print("Не удалось запустить сервис политики RDK:", result.stderr.strip() or result.stdout.strip())
        return False
    print("Сервис политики задач RDK переключён:", BPU_TASK, "->", RDK_HBM)
    return True

if not BPU_TASK:
    print("Для этой задачи нет проверяемого HBM; видео другой задачи вместо неё использоваться не будет.")
elif BPU_HBM is None or not BPU_HBM.is_file():
    print("HBM на Ubuntu не существует:", BPU_HBM)
elif not BPU_VIDEO_SCRIPT.is_file():
    print("Не найден записывающий скрипт замкнутого цикла на плате:", BPU_VIDEO_SCRIPT)
elif BPU_TASK == "ball_balance" and (MOTRIX_PYTHON is None or not MOTRIX_PYTHON.is_file()):
    print("Не найдено окружение Python MotrixLab:", MOTRIX_PYTHON)
elif BPU_TASK == "ball_balance" and (MOTRIX_RUNNER is None or not MOTRIX_RUNNER.is_file()):
    print("Не найден запускатель MotrixLab:", MOTRIX_RUNNER)
elif restart_rdk_policy_server():
    BPU_VIDEO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if BPU_TASK == "ball_balance":
        video_cmd = [str(MOTRIX_PYTHON), str(MOTRIX_RUNNER), "video", str(BPU_VIDEO_SCRIPT)]
    else:
        video_cmd = [sys.executable, str(BPU_VIDEO_SCRIPT)]
    if BPU_TASK != "ball_balance":
        video_cmd += ["--task", BPU_TASK]
    video_cmd += [
        "--bpu-host", RDK_HOST, "--bpu-port", str(BPU_PORT),
        "--hbm", str(BPU_HBM), "--output", str(BPU_VIDEO_OUTPUT),
        "--steps", str(BPU_VIDEO_STEPS), "--width", str(BPU_VIDEO_WIDTH),
        "--height", str(BPU_VIDEO_HEIGHT),
    ]
    print("Запуск генерации видео замкнутого цикла на плате:", " ".join(shlex.quote(x) for x in video_cmd))
    video_result = subprocess.run(
        video_cmd, cwd=PLAYGROUND_ROOT, env=os.environ.copy(),
        capture_output=True, text=True,
    )
    print("\\n".join((video_result.stdout + "\\n" + video_result.stderr).splitlines()[-35:]))
    if video_result.returncode != 0 or not BPU_VIDEO_OUTPUT.is_file():
        raise RuntimeError("Генерация видео BPU-in-the-loop не удалась; сначала посмотрите последние 35 строк лога выше.")
    BPU_REPORT = BPU_VIDEO_OUTPUT.with_suffix(".json")
    print("PASS-BPU-VIDEO:", BPU_VIDEO_OUTPUT)
    if BPU_REPORT.is_file():
        print(json.loads(BPU_REPORT.read_text(encoding="utf-8")))
    from IPython.display import Video, display
    display(Video(str(BPU_VIDEO_OUTPUT), embed=True))

SIM_COMMAND = {sim_command!r}
print("Шаблон команды воспроизведения MuJoCo:", SIM_COMMAND)
'''),
        md("""
## 6. Аудит RDK X5 и BPU

В предыдущей ячейке уже создано видео замкнутого цикла для задачи. Эта ячейка только проверяет связь с платой, метаданные HBM и логи сервиса; общий образец MobileNet доказывает лишь работоспособность рантайма системы и не может заменить видео задачи.
"""),
        code(r'''
import time

RDK_HOST = os.getenv("RDK_HOST", "192.168.8.128")
BPU_SMOKE_MODEL = os.getenv(
    "RDK_BPU_SMOKE_MODEL",
    "/opt/tros/humble/lib/dnn_benchmark_example/config/X5/mobilenetv1_224x224_nv12.bin",
)
BPU_SMOKE_INPUT_BYTES = int(os.getenv("RDK_BPU_SMOKE_INPUT_BYTES", "75264"))
BPU_HBM = os.getenv("RDK_BPU_HBM", "")
print("Образец BPU по умолчанию:", BPU_SMOKE_MODEL)
print("Входные данные образца BPU по умолчанию:", BPU_SMOKE_INPUT_BYTES, "байт uint8")
print("HBM политики MicroDuck:", BPU_HBM or "для этой задачи не предоставлен")
probe_script = "for x in hrt_model_exec hb_mapper hrt_bin_dump hrt_bin_info python3; do if command -v $x >/dev/null 2>&1; then echo $x=$(command -v $x); else echo $x=MISSING; fi; done"
remote = "sunrise@" + RDK_HOST
probe = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", remote, "bash -lc " + shlex.quote(probe_script)], capture_output=True, text=True)
if probe.returncode:
    print("RDK пока недоступен, приёмка на плате пропущена:", probe.stderr.strip() or "ssh failed")
else:
    print("RDK подключён:", RDK_HOST)
    print(probe.stdout)
    if not BPU_HBM:
        print("Нет HBM задачи: общий образец вместо инференса задачи не выполняем.")
        smoke_model = BPU_SMOKE_MODEL
        smoke_bytes = BPU_SMOKE_INPUT_BYTES
        smoke_info_cmd = "hrt_model_exec model_info --model_file " + shlex.quote(smoke_model)
        smoke_info = subprocess.run(["ssh", remote, "bash -lc " + shlex.quote(smoke_info_cmd)], capture_output=True, text=True)
        print("Код выхода model_info образца BPU:", smoke_info.returncode)
        print(smoke_info.stdout[-4000:])
        print(smoke_info.stderr[-1000:])
        if smoke_info.returncode == 0:
            import numpy as np
            local_input = OUTPUT_ROOT / "rdk_bpu_runtime_smoke.bin"
            np.zeros(smoke_bytes, dtype=np.uint8).tofile(local_input)
            remote_dir = "/tmp/microduck_notebook"
            remote_input = remote_dir + "/rdk_bpu_runtime_smoke.bin"
            subprocess.run(["ssh", remote, "mkdir", "-p", remote_dir], check=True)
            subprocess.run(["scp", str(local_input), remote + ":" + remote_input], check=True)
            dump_dir = remote_dir + "/runtime_dump"
            infer_cmd = "rm -rf " + shlex.quote(dump_dir) + " && mkdir -p " + shlex.quote(dump_dir) + " && hrt_model_exec infer --model_file " + shlex.quote(smoke_model) + " --input_file " + shlex.quote(remote_input) + " --enable_dump true --dump_format txt --dump_path " + shlex.quote(dump_dir)
            started = time.perf_counter()
            infer = subprocess.run(["ssh", remote, "bash -lc " + shlex.quote(infer_cmd)], capture_output=True, text=True)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print("BPU runtime sample exit:", infer.returncode, "round-trip ms:", round(elapsed_ms, 2))
            print(infer.stdout[-4000:])
            print(infer.stderr[-1000:])
            if infer.returncode == 0:
                print("PASS-rdk-bpu-runtime-sample: инференс образца X5 BPU вернул 0.")
    else:
        info_cmd = "hrt_model_exec model_info --model_file " + shlex.quote(BPU_HBM)
        info = subprocess.run(["ssh", remote, "bash -lc " + shlex.quote(info_cmd)], capture_output=True, text=True)
        print("HBM model_info exit:", info.returncode)
        print(info.stdout[-6000:])
        print(info.stderr[-2000:])
        if info.returncode == 0:
            # hrt_model_exec consumes binary tensors; this zero observation is
            # only a transport/BPU smoke input, not a task-quality evaluation.
            import numpy as np
            local_input = OUTPUT_ROOT / "zero_obs_61_f32.bin"
            np.zeros((1, 61), dtype=np.float32).tofile(local_input)
            remote_dir = "/tmp/microduck_notebook"
            remote_input = remote_dir + "/zero_obs_61_f32.bin"
            subprocess.run(["ssh", remote, "mkdir", "-p", remote_dir], check=True)
            subprocess.run(["scp", str(local_input), remote + ":" + remote_input], check=True)
            dump_dir = remote_dir + "/dump"
            infer_cmd = "rm -rf " + shlex.quote(dump_dir) + " && mkdir -p " + shlex.quote(dump_dir) + " && hrt_model_exec infer --model_file " + shlex.quote(BPU_HBM) + " --input_file " + shlex.quote(remote_input) + " --enable_dump true --dump_format txt --dump_path " + shlex.quote(dump_dir)
            started = time.perf_counter()
            infer = subprocess.run(["ssh", remote, "bash -lc " + shlex.quote(infer_cmd)], capture_output=True, text=True)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print("BPU infer exit:", infer.returncode, "round-trip ms:", round(elapsed_ms, 2))
            print(infer.stdout[-6000:])
            print(infer.stderr[-2000:])
            if infer.returncode == 0:
                print("PASS-rdk-bpu: hrt_model_exec infer вернул 0.")
            else:
                print("Не пройдено: ориентируйтесь в первую очередь на отчёт BPU-in-the-loop из раздела 5.")
'''),
        md("""
## 7. Выводы для стрима

- `PASS-local-onnx`: проверка ONNX checker и локального ONNX Runtime пройдена.
- `PASS-BPU-VIDEO`: каждое действие в видео Ubuntu MuJoCo вычислено на RDK X5 соответствующим HBM.
- `PASS-rdk-bpu-runtime-sample`: инференс установленного на плате образца X5 BPU завершился с кодом 0.
- `PASS-rdk-bpu`: SSH работает, на плате явно выбран BPU, команда инференса MicroDuck HBM завершилась с кодом 0, записаны формы входа и выхода и задержка.

Задача без `PASS-BPU-VIDEO` может считаться лишь «имеющимися материалами или описанием интерфейса», но не завершённым видео на плате.
"""),
    ]


TASKS = [
    ("01-basketball-balance-ppo-onnx-bpu-mujoco.ipynb", "Баскетбольный баланс / PPO", "Mjlab-Basketball-MicroDuck", None, "", False, None, None, "basketball"),
    ("02-browser-physical-perturbation-replay-and-interface.ipynb", "Физические возмущения в браузере / воспроизведение и интерфейс", None, None, "python serve_mjswan.py (из 01-task-information/02-mjswan-microduck-browser-physical-disturbance)", False, None, None, "perturbation"),
    ("03-stilt-walking-curriculum-and-onnx-bpu.ipynb", "Ходьба на ходулях / учебный план и формы", "Mjlab-Stilt-Flat-MicroDuck", None, "uv run python scripts/infer_policy.py --walking <STILT.onnx> --new-cmd-obs", False, None, None, "stilt"),
    ("04-swing-rotation-onnx-bpu-mujoco.ipynb", "Качание и вращение / самовозбуждаемые колебания", "Mjlab-SwingPump-MicroDuck", "01-任务资料/04-MicroDuck摆动旋转强化学习复现/assets/microduck_swing_alpha050.onnx", "uv run python scripts/infer_policy.py --walking outputs/policy.onnx", False, None, None, "swing"),
    ("05-ball-balance-fastsac-onnx-bpu.ipynb", "Баланс на шаре / FastSAC", "microduck-ball-balance", None, "python scripts/export_onnx.py run_dir=<FASTSAC_RUN_DIR> output=outputs/motrix_ball_balance_latest.onnx opset=11", False, None, None, "ball_balance"),
    ("06-ladder-climbing-contact-and-deployment-template.ipynb", "Подъём по лестнице / контактный курс", "Mjlab-Video-Ladder-Footstep-MicroDuck", None, "uv run python scripts/infer_policy.py --walking <LADDER.onnx> --new-cmd-obs", False, None, None, "ladder"),
    ("07-rdk-web-and-multi-policy-bpu-acceptance.ipynb", "Навигация / веб-интерфейс RDK / развёртывание нескольких политик", "Mjlab-Velocity-Flat-MicroDuck", None, "uv run python scripts/infer_policy.py --walking <WALKING.onnx> --new-cmd-obs", True, None, None, "walking"),
]


def build(filename: str, title: str, task_id: str | None, model_hint: str | None, sim_command: str, direct_connect: bool, video_hint: str | None, keyframe_hint: str | None, bpu_task: str | None):
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}
    notebook.cells = [
        md(f"# Notebook прямого эфира MicroDuck: {title}\n\nОбъединяет задачу, политику, ONNX, MuJoCo и приёмку RDK X5/BPU в единый воспроизводимый процесс."),
        md("## 0. Инструкция по запуску\n\nРекомендуется запускать Jupyter в Python-окружении `microduck-playground-stilts` (в исходном репозитории это каталог `02-可运行代码/microduck-playground-stilts`). Обученные модели и HBM для платы в Git не коммитятся; передаются через `MICRODUCK_ONNX`, `RDK_BPU_HBM` и `RDK_HOST`."),
        code(SETUP),
    ] + task_cells(model_hint, task_id, sim_command, direct_connect=direct_connect, video_hint=video_hint, keyframe_hint=keyframe_hint, bpu_task=bpu_task)
    nbf.write(notebook, ROOT / filename)


for item in TASKS:
    build(*item)
print(f"generated {len(TASKS)} notebooks in {ROOT}")
