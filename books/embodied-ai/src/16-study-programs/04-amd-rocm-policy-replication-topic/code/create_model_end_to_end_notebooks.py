#!/usr/bin/env python3
"""Create clean per-model end-to-end notebooks for the AMD ROCm topic."""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"


METADATA = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    },
}


SETUP = r'''
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import sys

try:
    from IPython.display import HTML, Markdown, display
except Exception:
    class Markdown(str):
        pass

    class HTML(str):
        pass

    def display(obj):
        print(obj)


def find_topic_root():
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "assets" / "metrics_snapshot.json").exists():
            return candidate
    raise RuntimeError("Запустите Jupyter из каталога темы AMD ROCm или из подкаталога notebooks.")


TOPIC_ROOT = find_topic_root()
ASSET_DIR = TOPIC_ROOT / "assets"
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/path/to/04mujoco-reproduction-act-pi0-smolvla"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/path/to/datasets/every_embodied"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/path/to/model/checkpoints"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", TOPIC_ROOT / "outputs"))

# The AMD teaching workflow should be runnable from local datasets/checkpoints.
# Avoid surprising network calls during class or when AUP/Radeon Cloud cannot
# reach Hugging Face.
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("NOTEBOOK_HF_OFFLINE", "1"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.environ.get("NOTEBOOK_HF_OFFLINE", "1"))
os.environ.setdefault("HF_DATASETS_OFFLINE", os.environ.get("NOTEBOOK_HF_OFFLINE", "1"))
os.environ.setdefault("HF_HOME", str(Path(os.environ.get("CACHE_ROOT", OUTPUT_ROOT / "cache")) / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))

def public_path(path):
    path = Path(path)
    replacements = [
        (TOPIC_ROOT, "$TOPIC_ROOT"),
        (PROJECT_ROOT, "$PROJECT_ROOT"),
        (DATA_ROOT, "$DATA_ROOT"),
        (MODEL_ROOT, "$MODEL_ROOT"),
        (OUTPUT_ROOT, "$OUTPUT_ROOT"),
    ]
    value = str(path)
    for root, label in sorted(replacements, key=lambda item: len(str(item[0])), reverse=True):
        root_value = str(root)
        if root_value and value.startswith(root_value):
            return label + value[len(root_value):]
    return value


print("TOPIC_ROOT = $TOPIC_ROOT")
print("PROJECT_ROOT =", public_path(PROJECT_ROOT))
print("DATA_ROOT =", public_path(DATA_ROOT))
print("MODEL_ROOT =", public_path(MODEL_ROOT))
print("OUTPUT_ROOT =", public_path(OUTPUT_ROOT))
'''


HELPERS = r'''
def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(public_path(x) if isinstance(x, (str, Path)) else str(x) for x in row) + " |")
    display(Markdown("\n".join(lines)))


def show_json(path, max_chars=5000):
    path = Path(path)
    if not path.exists():
        print("Файл не существует:", public_path(path))
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(data, ensure_ascii=False, indent=2)
    print(text[:max_chars] + ("\n..." if len(text) > max_chars else ""))
    return data


def show_video(filename, title):
    path = ASSET_DIR / filename
    display(Markdown(f"**{title}**"))
    if path.exists():
        rel = f"../assets/{filename}"
        display(HTML(f"<video controls muted preload='metadata' width='960'><source src='{rel}' type='video/mp4'></video>"))
    else:
        print("Нет видеоматериала:", public_path(path))


def show_image(filename, title, width=960):
    path = ASSET_DIR / filename
    display(Markdown(f"**{title}**"))
    if path.exists():
        rel = f"../assets/{filename}"
        display(HTML(f"<img src='{rel}' width='{width}'>"))
    else:
        print("Нет изображения:", public_path(path))


def run_cmd_preview(command, cwd=None):
    shown = [public_path(x) if isinstance(x, (str, Path)) else x for x in command]
    print("$", shlex.join([str(x) for x in shown]))
    if cwd:
        print("cwd =", public_path(cwd))


def tail_log(log_path, lines=40):
    path = Path(log_path)
    if not path.exists():
        print("Лог не существует:", public_path(path))
        return
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(content[-lines:]))


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


RUN_SMOKE = env_flag("RUN_SMOKE")
RUN_LONG_TRAIN = env_flag("RUN_LONG_TRAIN")
RUN_EVAL = env_flag("RUN_EVAL")
EVAL_SCRIPT = Path(os.environ.get("EVAL_SCRIPT", PROJECT_ROOT / "eval_policy_success.py"))


_XVFB_PROCESS = None


def ensure_xvfb_display():
    """Start a lightweight virtual display for headless MuJoCo evaluation."""
    global _XVFB_PROCESS
    if os.environ.get("DISPLAY"):
        print("DISPLAY =", os.environ["DISPLAY"])
        return None
    xvfb_bin = shutil.which("Xvfb")
    if not xvfb_bin:
        print("Xvfb не найден; при ошибке GLFW DISPLAY сначала установите xvfb.")
        return None
    display_id = os.environ.get("NOTEBOOK_XVFB_DISPLAY", ":99")
    _XVFB_PROCESS = subprocess.Popen(
        [xvfb_bin, display_id, "-screen", "0", "1280x1024x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = display_id
    print("Внутренний Xvfb Notebook запущен: DISPLAY =", display_id)
    return _XVFB_PROCESS


def ensure_project_layout():
    required = [PROJECT_ROOT / "asset" / "example_scene_y2.xml", PROJECT_ROOT / "mujoco_env"]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Текущий PROJECT_ROOT ещё не является рабочим проектом, отсутствует:")
        for path in missing:
            print(" -", public_path(path))
        print("Сначала задайте PROJECT_ROOT, затем запускайте ячейки обучения или оценки.")
        return False
    return True


def write_json_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    except Exception:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    print("Конфигурация записана:", public_path(path))
    return path


def make_lerobot_train_config(policy_type, dataset_repo_id, dataset_root, output_dir, steps, batch_size, chunk_size, n_action_steps, seed=42):
    save_freq = int(os.environ.get(f"{policy_type.upper()}_SAVE_FREQ", os.environ.get("SAVE_FREQ", str(steps))))
    return {
        "dataset": {
            "repo_id": dataset_repo_id,
            "root": str(dataset_root),
            "use_imagenet_stats": True,
        },
        "policy": {
            "type": policy_type,
            "chunk_size": int(chunk_size),
            "n_action_steps": int(n_action_steps),
            "device": "cuda",
        },
        "output_dir": str(output_dir),
        "job_name": Path(output_dir).name,
        "batch_size": int(batch_size),
        "steps": int(steps),
        "save_freq": max(1, save_freq),
        "log_freq": 20,
        "num_workers": 4,
        "seed": int(seed),
        "resume": False,
        "eval_freq": -1,
        "save_checkpoint": True,
        "use_policy_training_preset": True,
        "wandb": {"enable": False, "disable_artifact": True},
    }


def train_lerobot_config_in_notebook(config_path, enabled=False, progress_name="train"):
    """Run LeRobot offline training directly inside the notebook kernel.

    The notebook cell owns dataset creation, policy creation, optimizer steps,
    checkpoint saving, tqdm progress, and metric JSONL writing.
    """
    config_path = Path(config_path)
    print("config =", public_path(config_path))
    if not enabled:
        print("Не запущено. После установки RUN_SMOKE=1 или RUN_LONG_TRAIN=1 эта ячейка будет обучать модель прямо в Notebook.")
        return None
    if not ensure_project_layout():
        return None

    import time
    from contextlib import nullcontext

    import draccus
    import torch
    from torch.amp import GradScaler
    from tqdm.auto import tqdm

    from lerobot.common.datasets.factory import make_dataset
    from lerobot.common.datasets.sampler import EpisodeAwareSampler
    from lerobot.common.optim.factory import make_optimizer_and_scheduler
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.policies.utils import get_device_from_parameters
    from lerobot.common.utils.random_utils import set_seed
    from lerobot.common.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint
    from lerobot.common.utils.utils import get_safe_torch_device
    from lerobot.configs.train import TrainPipelineConfig

    cfg = draccus.parse(TrainPipelineConfig, config_path=config_path, args=[])
    cfg.validate()
    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Creating dataset...")
    dataset = make_dataset(cfg)
    print("Creating policy...")
    pretrained_override = os.environ.get(f"{cfg.policy.type.upper()}_PRETRAINED_PATH_OVERRIDE") or os.environ.get("POLICY_PRETRAINED_PATH_OVERRIDE")
    if pretrained_override and not cfg.resume:
        cfg.policy.pretrained_path = str(Path(pretrained_override))
        print("pretrained override =", public_path(cfg.policy.pretrained_path))
    elif cfg.policy.type == "pi0" and not cfg.resume:
        cfg.policy.pretrained_path = "lerobot/pi0"
    elif cfg.policy.type == "smolvla" and not cfg.resume:
        smolvla_base_candidates = [
            os.environ.get("SMOLVLA_BASE_PATH"),
            os.environ.get("SMOLVLA_PRETRAINED_BASE_PATH"),
            str(MODEL_ROOT / "smolvla_base" / "pretrained_model"),
            str(MODEL_ROOT / "lerobot_smolvla_base_legacy"),
            str(MODEL_ROOT / "lerobot_smolvla_base"),
        ]
        local_smolvla_base = next((Path(p) for p in smolvla_base_candidates if p and Path(p).exists()), None)
        if local_smolvla_base is not None:
            cfg.policy.pretrained_path = str(local_smolvla_base)
            print("local smolvla base =", public_path(cfg.policy.pretrained_path))
        else:
            cfg.policy.pretrained_path = "lerobot/smolvla_base"
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    # Compatibility for newer Transformers: PaliGemmaForConditionalGeneration may expose
    # language_model as GemmaModel directly, while this LeRobot Pi0 code expects
    # language_model.model.  Use a non-Module proxy so checkpoints/state_dict stay clean.
    if cfg.policy.type == "pi0":
        try:
            lm = policy.model.paligemma_with_expert.paligemma.language_model
            if not hasattr(lm, "model"):
                class _LanguageModelCoreProxy:
                    def __init__(self, core):
                        self._core = core

                    def __getattr__(self, name):
                        return getattr(self._core, name)

                object.__setattr__(lm, "model", _LanguageModelCoreProxy(lm))
                print("patched Pi0 PaliGemma language_model.model compatibility proxy")
        except Exception as exc:
            print(f"Pi0 PaliGemma compatibility patch skipped: {exc}")

    policy.to(device)
    policy.train()

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    def _dataset_column_values(name):
        hf_dataset = getattr(dataset, "hf_dataset", None)
        if hf_dataset is None or name not in getattr(hf_dataset, "column_names", []):
            return None
        values = hf_dataset[name]
        try:
            return list(values)
        except TypeError:
            return [values[i] for i in range(len(values))]

    def _task_name_map():
        meta = getattr(dataset, "meta", None)
        tasks = getattr(meta, "tasks", None)
        if tasks is None:
            return {}
        if isinstance(tasks, dict):
            return {int(k): str(v) for k, v in tasks.items()}
        try:
            return {int(k): str(v) for k, v in dict(tasks).items()}
        except Exception:
            return {}

    def _make_weighted_sampler(generator):
        mode = os.environ.get("NOTEBOOK_FRAME_WEIGHT_MODE", "").strip().lower()
        if not mode or mode in {"0", "none", "off", "false"}:
            return None, {"mode": "none"}
        weights = torch.ones(len(dataset), dtype=torch.double)
        info = {"mode": mode, "num_frames": len(dataset)}

        if "blue" in mode:
            blue_weight = float(os.environ.get("NOTEBOOK_BLUE_WEIGHT", "2.0"))
            mask = [False] * len(dataset)
            task_indices = _dataset_column_values("task_index")
            task_names = _task_name_map()
            if task_indices is not None and task_names:
                for idx, task_index in enumerate(task_indices):
                    task_text = task_names.get(int(task_index), "").lower()
                    mask[idx] = ("blue" in task_text) or ("蓝" in task_text)  # "蓝" — «синий»: сопоставление с китайской инструкцией, не переводить
            else:
                for column in ["task", "language_instruction", "instruction"]:
                    values = _dataset_column_values(column)
                    if values is None:
                        continue
                    for idx, value in enumerate(values):
                        text = str(value).lower()
                        mask[idx] = ("blue" in text) or ("蓝" in text)  # "蓝" — «синий»: сопоставление с китайской инструкцией, не переводить
                    break
            blue_count = int(sum(mask))
            if blue_count == 0:
                print("Предупреждение: NOTEBOOK_FRAME_WEIGHT_MODE=blue, но кадры с инструкциями blue или 蓝 не найдены, выборка возвращается к равномерной.")
            else:
                for idx, is_blue in enumerate(mask):
                    if is_blue:
                        weights[idx] *= blue_weight
            info.update({"blue_weight": blue_weight, "blue_frames": blue_count})

        weight_file = os.environ.get("NOTEBOOK_FRAME_WEIGHT_JSON")
        if weight_file:
            payload = json.loads(Path(weight_file).read_text(encoding="utf-8"))
            for key, value in payload.items():
                weights[int(key)] *= float(value)
            info.update({"weight_json": public_path(weight_file), "json_entries": len(payload)})

        if float(weights.sum()) <= 0:
            raise ValueError("Сумма весов выборки равна 0.")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        info.update(
            {
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
                "weight_mean": float(weights.mean()),
            }
        )
        return sampler, info

    generator = torch.Generator()
    if cfg.seed is not None:
        generator.manual_seed(int(cfg.seed))

    weighted_sampler, sampler_info = _make_weighted_sampler(generator)
    if weighted_sampler is not None:
        shuffle = False
        sampler = weighted_sampler
        print("Notebook weighted sampler =", json.dumps(sampler_info, ensure_ascii=False))
    elif hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator if sampler is None else None,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    # Do not use itertools.cycle here: it caches every batch and can exhaust
    # host RAM during a long Notebook training run. Recreate the iterator only
    # when the finite DataLoader is exhausted.
    dl_iter = iter(dataloader)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "notebook_train_metrics.jsonl"
    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    print(f"output_dir = {public_path(output_dir)}")
    print(f"steps = {cfg.steps}, batch_size = {cfg.batch_size}, frames = {dataset.num_frames}, episodes = {dataset.num_episodes}")
    print(f"learnable_params = {num_learnable:,}, total_params = {num_total:,}")

    last_metrics = None
    progress = tqdm(range(1, cfg.steps + 1), desc=progress_name, dynamic_ncols=True)
    start_all = time.perf_counter()
    for step in progress:
        load_start = time.perf_counter()
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)
        data_s = time.perf_counter() - load_start
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)

        update_start = time.perf_counter()
        device_from_policy = get_device_from_parameters(policy)
        with torch.autocast(device_type=device_from_policy.type) if cfg.policy.use_amp else nullcontext():
            loss, output_dict = policy.forward(batch)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            cfg.optimizer.grad_clip_norm,
            error_if_nonfinite=False,
        )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if hasattr(policy, "update"):
            policy.update()
        update_s = time.perf_counter() - update_start

        is_log_step = cfg.log_freq > 0 and (step % cfg.log_freq == 0 or step == 1 or step == cfg.steps)
        is_saving_step = cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps)
        if is_log_step:
            last_metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "update_s": float(update_s),
                "data_s": float(data_s),
                "elapsed_s": float(time.perf_counter() - start_all),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last_metrics, ensure_ascii=False) + "\n")
            progress.set_postfix(
                loss=f"{last_metrics['loss']:.4f}",
                lr=f"{last_metrics['lr']:.1e}",
                updt_s=f"{last_metrics['update_s']:.3f}",
            )
        if is_saving_step:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            print(f"\nSaving checkpoint at step {step}: {public_path(checkpoint_dir)}")
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)

    print("Обучение завершено. metrics =", public_path(metrics_path))
    if last_metrics is not None:
        print(json.dumps(last_metrics, ensure_ascii=False, indent=2))
    return {"output_dir": output_dir, "metrics_path": metrics_path, "last_metrics": last_metrics}


def load_eval_module():
    import importlib.util

    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Скрипт оценки не существует: {public_path(EVAL_SCRIPT)}")
    spec = importlib.util.spec_from_file_location("notebook_eval_policy_success", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_eval_policy_in_notebook(
    policy_name,
    policy_path,
    result_path,
    episodes,
    seed_start,
    render=False,
    enabled=False,
    repo_id=None,
    dataset_root=None,
):
    print("policy =", policy_name)
    print("policy_path =", public_path(policy_path))
    print("result =", public_path(result_path))
    if not enabled:
        print("Не запущено. После установки RUN_EVAL=1 эта ячейка загрузит политику и выполнит замкнутую оценку прямо в Notebook.")
        return None
    if not ensure_project_layout():
        return None

    import argparse
    from contextlib import contextmanager
    from tqdm.auto import tqdm

    @contextmanager
    def pushd(path):
        old = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old)

    ensure_xvfb_display()
    module = load_eval_module()
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()

    args = argparse.Namespace(
        policy=policy_name,
        episodes=int(episodes),
        seed_start=int(seed_start),
        max_action_steps=int(os.environ.get("EVAL_MAX_ACTION_STEPS", "400")),
        hz=float(os.environ.get("EVAL_HZ", "20")),
        render=bool(render),
        output_jsonl=result_path,
        device=os.environ.get("EVAL_DEVICE", "cuda"),
        reset_policy_each_action=env_flag("EVAL_RESET_POLICY_EACH_ACTION", False),
        act_n_action_steps=None,
        act_force_dataset_gripper=False,
        act_clamp_timestamp=False,
        act_policy_path=Path(policy_path),
        act_repo_id=repo_id or "datawhale_eai_pnp",
        act_dataset_root=Path(dataset_root or "./demo_data"),
        act_episode_timestamp_offsets="",
        act_episode_source_flags="",
        physical_success=env_flag("EVAL_PHYSICAL_SUCCESS", True),
        physical_min_lift=float(os.environ.get("EVAL_PHYSICAL_MIN_LIFT", "0.06")),
        physical_min_lift_steps=int(os.environ.get("EVAL_PHYSICAL_MIN_LIFT_STEPS", "3")),
        physical_final_upright_cos=float(os.environ.get("EVAL_PHYSICAL_FINAL_UPRIGHT_COS", "0.85")),
        smolvla_policy_path=Path(policy_path),
        pi0_policy_path=Path(policy_path),
        pi0_repo_id=repo_id or os.environ.get("PI0_DATASET_REPO_ID", "datawhale_eai_pnp_language"),
        pi0_dataset_root=Path(dataset_root or os.environ.get("PI0_DATASET_ROOT", "./demo_data_language")),
    )

    with pushd(PROJECT_ROOT):
        if policy_name == "act":
            policy = module.make_act_policy(
                args.device,
                args.act_policy_path,
                args.act_repo_id,
                args.act_dataset_root,
                n_action_steps=args.act_n_action_steps,
                episode_timestamp_offsets=args.act_episode_timestamp_offsets,
                episode_source_flags=args.act_episode_source_flags,
            )
            rollout = module.rollout_act
        elif policy_name == "smolvla":
            policy = module.make_smolvla_policy(args.device, args.smolvla_policy_path)
            rollout = module.rollout_language_policy
        elif policy_name == "pi0":
            policy = module.make_pi0_policy(args.device, args.pi0_policy_path, args.pi0_repo_id, args.pi0_dataset_root)
            rollout = module.rollout_language_policy
        else:
            raise ValueError(policy_name)

        rows = []
        for offset in tqdm(range(args.episodes), desc=f"{policy_name} eval", dynamic_ncols=True):
            seed = args.seed_start + offset
            row = rollout(args, policy, seed)
            rows.append(row)
            with result_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False))
    summarize_jsonl(result_path)
    return rows


def list_checkpoints(run_dir):
    run_dir = Path(run_dir)
    candidates = []
    for pattern in ["checkpoints/*/pretrained_model", "checkpoint*/pretrained_model", "*/pretrained_model", "pretrained_model"]:
        candidates.extend(run_dir.glob(pattern))
    unique = sorted(set(candidates))
    if not unique:
        print("Checkpoint пока не найдены:", public_path(run_dir))
        return []
    for path in unique:
        print(" -", public_path(path))
    return unique


def resolve_eval_policy(default_path, trained_run_dir=None, env_name=None):
    if env_name and os.environ.get(env_name):
        path = Path(os.environ[env_name])
        print("Оценка использует веса, заданные переменной окружения:", public_path(path))
        return path
    if trained_run_dir is not None and env_flag("EVAL_USE_LONG_TRAIN"):
        checkpoints = list_checkpoints(trained_run_dir)
        if checkpoints:
            path = checkpoints[-1]
            print("Оценка использует последний checkpoint текущего длительного обучения:", public_path(path))
            return path
        print("Checkpoint текущего длительного обучения не найден, откат к защищённым весам.")
    path = Path(default_path)
    print("Оценка использует защищённые или предобученные веса:", public_path(path))
    return path


def summarize_jsonl(path):
    path = Path(path)
    if not path.exists():
        print("Файл результатов JSONL ещё не существует:", public_path(path))
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    legacy = sum(bool(row.get("success") or row.get("legacy_success")) for row in rows)
    if rows and all("physical_success" in row for row in rows):
        physical_count = sum(bool(row.get("physical_success")) for row in rows)
        physical_text = str(physical_count) + "/" + str(total)
    else:
        physical_text = "не записано"
    md_table(
        ["Файл результатов", "episodes", "legacy_success", "physical_success"],
        [(public_path(path), total, f"{legacy}/{total}", physical_text)],
    )
    return rows
'''


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str, output: str | None = None) -> dict:
    outputs = []
    execution_count = None
    if output is not None:
        execution_count = 1
        outputs = [{"name": "stdout", "output_type": "stream", "text": output}]
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": source.strip("\n") + "\n",
    }


def write_nb(filename: str, cells: list[dict]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / filename
    nb = {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


def smolvla_cells() -> list[dict]:
    return [
        md(
            """
            # 14 SmolVLA сквозной цикл: обучение, оценка, видео и диагностика

            Этот Notebook объединяет полный путь обучения SmolVLA в одном файле: сначала смотрим веса и видео, которые уже удалось воспроизвести, затем проверяем данные и даём команды для smoke, длительного обучения, строгой оценки и отслеживания логов.

            Рекомендуемый порядок запуска: для первой демонстрации на занятии включайте только `RUN_SMOKE=1 RUN_EVAL=1`; для полного воспроизведения эксперимента включайте `RUN_LONG_TRAIN=1`. Вывод длительного обучения записывается прямо в ячейки Notebook, а не вставляется статичным логом.
            """
        ),
        code(SETUP),
        code(HELPERS),
        md("## Checkpoint 1: сначала поймём, почему эта версия выбрана основной"),
        code(
            r'''
rows = [
    ("Запись исторического учебника", "53/60", "SmolVLA weighted step500, красная и синяя чашки сбалансированы лучше"),
    ("Текущий результат воспроизведения", "57/60", "red 27/30, blue 30/30"),
    ("Рекомендация к публикации", "основной вес", "подходит как предпросмотр без обучения и кейс Notebook по умолчанию"),
]
md_table(["Пункт", "Результат", "Пояснение"], rows)
''',
            "Пункт | Результат | Пояснение\nЗапись исторического учебника | 53/60 | SmolVLA weighted step500, красная и синяя чашки сбалансированы лучше\nТекущий результат воспроизведения | 57/60 | red 27/30, blue 30/30\nРекомендация к публикации | основной вес | подходит как предпросмотр без обучения и кейс Notebook по умолчанию\n",
        ),
        md("## Checkpoint 2: строгие успешные и неуспешные видео"),
        code(
            r'''
show_video("smolvla_weighted500_red_success_seed0.mp4", "Успешное воспроизведение с красной чашкой: weighted500 seed0")
show_video("smolvla_weighted500_blue_success_seed0.mp4", "Успешное воспроизведение с синей чашкой: weighted500 seed0")
show_video("smolvla_weighted500_red_failure_seed8.mp4", "Неуспешное воспроизведение с красной чашкой: для наблюдения проблем upright и release")
''',
        ),
        md(
            """
            ## Checkpoint 3: проверка путей к данным и весам

            Ожидаемое время: несколько секунд. Обучения здесь нет: только проверяем, что данные, веса и выходной каталог указывают на правильные места.
            """
        ),
        code(
            r'''
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language")
TRAIN_DATA_ROOT = Path(os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "demo_data_language"))
PRETRAINED_POLICY = Path(os.environ.get("SMOLVLA_POLICY_PATH", MODEL_ROOT / "smolvla_weighted_000500" / "pretrained_model"))

rows = [
    ("DATASET_REPO_ID", DATASET_REPO_ID),
    ("TRAIN_DATA_ROOT", TRAIN_DATA_ROOT),
    ("SMOLVLA_POLICY_PATH", PRETRAINED_POLICY),
    ("Метаданные данных", TRAIN_DATA_ROOT / "meta" / "info.json"),
]
md_table(["Переменная", "Текущее значение"], rows)
''',
        ),
        md(
            """
            ## Checkpoint 4: генерация конфигурации и реальный запуск smoke или длительного обучения

            Ожидаемое время: smoke около 1–5 минут; длительное обучение с `SMOLVLA_STEPS=5000` обычно занимает от десятков минут до нескольких часов в зависимости от ROCm, batch size и скорости диска с данными.
            Эта ячейка — настоящая точка входа в обучение: после `RUN_SMOKE=1` или `RUN_LONG_TRAIN=1` она сама создаёт dataset, policy, optimizer и цикл обучения внутри ядра Notebook и показывает прогресс tqdm.
            """
        ),
        code(
            r'''
CONFIG_DIR = OUTPUT_ROOT / "configs"
LOG_DIR = OUTPUT_ROOT / "logs"
RUN_ROOT = OUTPUT_ROOT / "runs" / "smolvla_weighted_repro"
SMOKE_OUTPUT = RUN_ROOT / "smoke"
LONG_OUTPUT = RUN_ROOT / "weighted_full"

smoke_config = make_lerobot_train_config(
    "smolvla", DATASET_REPO_ID, TRAIN_DATA_ROOT, SMOKE_OUTPUT,
    steps=2, batch_size=2, chunk_size=50, n_action_steps=50,
)
long_config = make_lerobot_train_config(
    "smolvla", DATASET_REPO_ID, TRAIN_DATA_ROOT, LONG_OUTPUT,
    steps=int(os.environ.get("SMOLVLA_STEPS", "5000")),
    batch_size=int(os.environ.get("SMOLVLA_BATCH_SIZE", "4")),
    chunk_size=50,
    n_action_steps=50,
)
smoke_config_path = write_json_yaml(CONFIG_DIR / "smolvla_smoke.yaml", smoke_config)
long_config_path = write_json_yaml(CONFIG_DIR / "smolvla_weighted_full.yaml", long_config)

train_lerobot_config_in_notebook(smoke_config_path, enabled=RUN_SMOKE, progress_name="SmolVLA smoke")
train_lerobot_config_in_notebook(long_config_path, enabled=RUN_LONG_TRAIN, progress_name="SmolVLA long train")
''',
            "Нативная точка входа обучения в Notebook. RUN_SMOKE/RUN_LONG_TRAIN по умолчанию выключены; включайте их после проверки путей.\n",
        ),
        md(
            """
            ## Checkpoint 5: просмотр логов обучения и checkpoint в реальном времени

            Ожидаемое время: несколько секунд. Во время длительного обучения эту ячейку можно запускать повторно и смотреть metrics, которые записал Notebook, и уже сохранённые checkpoint.
            """
        ),
        code(
            r'''
print("smoke metrics:")
tail_log(SMOKE_OUTPUT / "notebook_train_metrics.jsonl", lines=20)
print("\nlong train metrics:")
tail_log(LONG_OUTPUT / "notebook_train_metrics.jsonl", lines=40)
print("\ncheckpoints:")
list_checkpoints(LONG_OUTPUT)
''',
        ),
        md(
            """
            ## Checkpoint 6: измеренное сравнение для уже завершённого длительного обучения

            Это сравнение видео и метрик уже завершённого и проверенного на устройстве AMD запуска. Оно помогает сначала увидеть, как выглядит правильный результат. Оно не заменяет вывод обучения в этом Notebook.
            """
        ),
        code(
            r'''
rows = [
    ("parent", "5000/5000 steps", "базовый SmolVLA сошёлся до пригодного checkpoint"),
    ("weighted-blue", "selected step 500/1000", "step500 сбалансированнее step1000"),
    ("strict gate", "57/60", "legacy 60/60, physical 57/60"),
]
md_table(["Этап", "Прогресс обучения или оценки", "Вывод"], rows)
''',
            "Этап | Прогресс обучения или оценки | Вывод\nparent | 5000/5000 steps | базовый SmolVLA сошёлся до пригодного checkpoint\nweighted-blue | selected step 500/1000 | step500 сбалансированнее step1000\nstrict gate | 57/60 | legacy 60/60, physical 57/60\n",
        ),
        code(
            r'''
show_image("training_progress_overview.png", "Историческая динамика обучения и результаты замкнутого цикла")
show_image("smolvla_red_blue_success.png", "Сравнение по инструкциям для красной и синей чашек")
''',
        ),
        md(
            """
            ## Checkpoint 7: строгая оценка внутри Notebook

            Ожидаемое время: 10 episode обычно занимают 10–30 минут; `SMOLVLA_EVAL_EPISODES=60` займёт дольше.
            Эта ячейка загружает политику в ядре Notebook и выполняет замкнутый rollout по каждому seed; если монитора нет, автоматически пробуется запуск Xvfb.
            """
        ),
        code(
            r'''
eval_episodes = os.environ.get("SMOLVLA_EVAL_EPISODES", "10")
eval_policy = resolve_eval_policy(PRETRAINED_POLICY, LONG_OUTPUT, "SMOLVLA_EVAL_POLICY_PATH")
run_eval_policy_in_notebook(
    "smolvla",
    eval_policy,
    OUTPUT_ROOT / f"smolvla_eval_seed0_{int(eval_episodes)-1}.jsonl",
    episodes=eval_episodes,
    seed_start=0,
    render=env_flag("RENDER_EVAL"),
    enabled=RUN_EVAL,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
summarize_jsonl(OUTPUT_ROOT / f"smolvla_eval_seed0_{int(eval_episodes)-1}.jsonl")
'''
        ),
        md("## Checkpoint 8: как читать результаты"),
        code(
            r'''
summary = {
    "candidate": "weighted_000500",
    "episodes": 60,
    "physical_success_count": 57,
    "legacy_success_count": 60,
    "by_color": {"red": "27/30", "blue": "30/30"},
    "release_decision": "использовать как предобученные веса учебника по умолчанию",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
''',
            '{\n  "candidate": "weighted_000500",\n  "episodes": 60,\n  "physical_success_count": 57,\n  "legacy_success_count": 60,\n  "by_color": {\n    "red": "27/30",\n    "blue": "30/30"\n  },\n  "release_decision": "использовать как предобученные веса учебника по умолчанию"\n}\n',
        ),
    ]


def pi0_cells() -> list[dict]:
    return [
        md(
            """
            # 15 Pi0 сквозной цикл: доступ, длительное обучение, строгий ввод и корзины провалов

            Эта версия Pi0 — не основная линия «полного воспроизведения исторического hard8», а продвинутая диагностическая модель. Notebook в основном показывает доступ gated, конфигурацию длительного обучения, checkpoint S8500, median reducer, строгую оценку и анализ корзин провалов.
            """
        ),
        code(SETUP),
        code(HELPERS),
        md("## Checkpoint 1: текущие границы восстановления"),
        code(
            r'''
rows = [
    ("unseen10", "9/10", "выше раннего результата 7/10"),
    ("full14", "12/14", "красная 6/7, синяя 6/7"),
    ("hard8", "6/8", "исторический 8/8 не восстановлен, нельзя писать как полное воспроизведение"),
]
md_table(["Панель", "Текущий результат", "Вывод"], rows)
''',
            "Панель | Текущий результат | Вывод\nunseen10 | 9/10 | выше раннего результата 7/10\nfull14 | 12/14 | красная 6/7, синяя 6/7\nhard8 | 6/8 | исторический 8/8 не восстановлен, нельзя писать как полное воспроизведение\n",
        ),
        md(
            """
            ## Checkpoint 2: проверка доступа gated и кэша

            Ожидаемое время: несколько секунд. Pi0 требует проверки доступа к модели, кэша Hugging Face и пути к обучающим данным; Notebook только проверяет состояние и не выводит никакие token.
            """
        ),
        code(
            r'''
print("HF_TOKEN передан:", bool(os.environ.get("HF_TOKEN")))
rows = [
    ("HF_TOKEN", "проверяется только наличие, token в Notebook не выводится"),
    ("HF_HOME", "задан" if os.environ.get("HF_HOME") else "не задан"),
    ("Кэш моделей", "рекомендуется хранить в $CACHE_ROOT/huggingface или на постоянном диске облачной платформы"),
]
md_table(["Пункт", "Пояснение"], rows)
''',
        ),
        md(
            """
            ## Checkpoint 3: генерация конфигурации и реальный запуск обучения или дообучения

            Ожидаемое время: smoke около 1–5 минут; `PI0_STEPS=8500` — длительное обучение, может занять несколько часов.
            Это настоящая точка входа в обучение: `RUN_SMOKE=1` проверяет окружение, а при `RUN_LONG_TRAIN=1` цикл обучения Pi0 запускается прямо в ядре Notebook с прогрессом tqdm.
            """
        ),
        code(
            r'''
DATASET_REPO_ID = "datawhale_eai_pnp_pi0_clean_oracle_y060z000_3100_3139_g8_rebuild_v1"
TRAIN_DATA_ROOT = Path(os.environ.get("PI0_TRAIN_DATA_ROOT", DATA_ROOT / "pi0_clean40_successonly"))
PI0_POLICY_PATH = Path(os.environ.get("PI0_POLICY_PATH", MODEL_ROOT / "pi0_clean40_successonly_blue2x_s8500" / "pretrained_model"))
CONFIG_DIR = OUTPUT_ROOT / "configs"
LOG_DIR = OUTPUT_ROOT / "logs"
RUN_ROOT = OUTPUT_ROOT / "runs" / "pi0_s8500_repro"
SMOKE_OUTPUT = RUN_ROOT / "smoke"
LONG_OUTPUT = RUN_ROOT / "clean40_blue2x_s8500"

smoke_config = make_lerobot_train_config(
    "pi0", DATASET_REPO_ID, TRAIN_DATA_ROOT, SMOKE_OUTPUT,
    steps=2, batch_size=1, chunk_size=50, n_action_steps=50,
)
long_config = make_lerobot_train_config(
    "pi0", DATASET_REPO_ID, TRAIN_DATA_ROOT, LONG_OUTPUT,
    steps=int(os.environ.get("PI0_STEPS", "8500")),
    batch_size=int(os.environ.get("PI0_BATCH_SIZE", "2")),
    chunk_size=50,
    n_action_steps=50,
)
smoke_config_path = write_json_yaml(CONFIG_DIR / "pi0_smoke.yaml", smoke_config)
long_config_path = write_json_yaml(CONFIG_DIR / "pi0_clean40_blue2x_s8500.yaml", long_config)

train_lerobot_config_in_notebook(smoke_config_path, enabled=RUN_SMOKE, progress_name="Pi0 smoke")
train_lerobot_config_in_notebook(long_config_path, enabled=RUN_LONG_TRAIN, progress_name="Pi0 long train")
''',
            "Нативная точка входа обучения в Notebook. RUN_SMOKE/RUN_LONG_TRAIN по умолчанию выключены; перед обучением Pi0 обязательно проверьте доступ gated.\n",
        ),
        md(
            """
            ## Checkpoint 4: просмотр логов обучения и checkpoint в реальном времени

            Ожидаемое время: несколько секунд. Во время обучения эту ячейку можно запускать повторно и следить за loss, step, точками сохранения и появлением OOM или NaN.
            """
        ),
        code(
            r'''
print("smoke metrics:")
tail_log(SMOKE_OUTPUT / "notebook_train_metrics.jsonl", lines=20)
print("\nlong train metrics:")
tail_log(LONG_OUTPUT / "notebook_train_metrics.jsonl", lines=40)
print("\ncheckpoints:")
list_checkpoints(LONG_OUTPUT)
''',
        ),
        md(
            """
            ## Checkpoint 5: измеренное сравнение для уже завершённого длительного обучения

            Это сравнение результатов, полученных ранее на устройстве AMD. Оно показывает, до какого уровня сейчас восстановлен Pi0; оно не заменяет реальный вывод этого запуска Notebook.
            """
        ),
        code(
            r'''
rows = [
    ("S7500", "canary2/4", "родительский checkpoint для продолжения обучения"),
    ("S8500", "canary3/4", "переход к расширенной оценке full14"),
    ("full14 median", "12/14", "используется как текущая защищённая версия"),
]
md_table(["Этап", "Порог", "Вывод"], rows)
''',
            "Этап | Порог | Вывод\nS7500 | canary2/4 | родительский checkpoint для продолжения обучения\nS8500 | canary3/4 | переход к расширенной оценке full14\nfull14 median | 12/14 | используется как текущая защищённая версия\n",
        ),
        md(
            """
            ## Checkpoint 6: строгая замкнутая оценка внутри Notebook

            Ожидаемое время: 14 episode обычно занимают несколько десятков минут; для быстрого порога можно использовать `PI0_EVAL_EPISODES=4`.
            Эта ячейка загружает Pi0 прямо в ядре Notebook и выполняет замкнутую оценку по протоколу `observation.state` размерности 8, соответствующему checkpoint.
            """
        ),
        code(
            r'''
eval_episodes = os.environ.get("PI0_EVAL_EPISODES", "14")
eval_policy = resolve_eval_policy(PI0_POLICY_PATH, LONG_OUTPUT, "PI0_EVAL_POLICY_PATH")
result_path = OUTPUT_ROOT / f"pi0_s8500_seed{os.environ.get('PI0_EVAL_SEED_START', '3000')}_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "pi0",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=os.environ.get("PI0_EVAL_SEED_START", "3000"),
    render=env_flag("RENDER_EVAL"),
    enabled=RUN_EVAL,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
summarize_jsonl(result_path)
''',
        ),
        md("## Checkpoint 7: корзины провалов и проверка видео"),
        code(
            r'''
summary = {
    "episodes": 14,
    "physical_success_count": 12,
    "by_color": {"blue": "6/7", "red": "6/7"},
    "failures": [
        "blue seed3007: lifted but upright unstable",
        "red seed3012: no enough lift/contact",
    ],
    "do_not_claim": "hard8 8/8 has not been recovered",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
show_video("pi0_ep2_raw_vs_finisher_side_by_side.mp4", "Видео сравнения Pi0 raw и диагностического finisher")
show_image("pi0_raw_vs_finisher_diagnostic.png", "Диагностический график Pi0: raw, scaffold и строгая оценка")
''',
            '{\n  "episodes": 14,\n  "physical_success_count": 12,\n  "by_color": {\n    "blue": "6/7",\n    "red": "6/7"\n  },\n  "failures": [\n    "blue seed3007: lifted but upright unstable",\n    "red seed3012: no enough lift/contact"\n  ],\n  "do_not_claim": "hard8 8/8 has not been recovered"\n}\n',
        ),
    ]


def act_cells() -> list[dict]:
    return [
        md(
            """
            # 16 ACT сквозной цикл: цепочка обучения и диагностика провалов

            Цепочка обучения, инференса и оценки ACT уже работает. Лучший защищённый кандидат после дообучения на AMD395 даёт `15/30` на строгих 30 episode, что выше fallback stable61 с `7/30`; обычный ACT на W7900 — `0/30`, старый артефакт DAgger — `2/30`. Значение `17/30` из старого учебника сохранено как историческая цель и не должно выдаваться за результат текущих весов.
            """
        ),
        code(SETUP),
        code(HELPERS),
        md("## Checkpoint 1: текущие и исторические результаты"),
        code(
            r'''
rows = [
    ("Запись исторического учебника", "17/30 (историческая цель)", "старая сводка DAgger; текущими защищёнными весами полностью не восстановлено"),
    ("Текущий защищённый кандидат repair15", "15/30", "AMD395; step1500; три панели 3/10 + 4/10 + 8/10"),
    ("Старый fallback stable61", "7/30", "AMD395; step2500; контрольный baseline"),
    ("Рекомендация к публикации", "можно защищать и использовать в учебных целях", "не утверждаем, что исторический 17/30 полностью восстановлен"),
]
md_table(["Пункт", "Результат", "Пояснение"], rows)
''',
            "Пункт | Результат | Пояснение\nЗапись исторического учебника | 17/30 | лучший результат старого DAgger\nЛучший результат текущего воспроизведения | 7/30 | stable61 step2500 strict30\nРекомендация к публикации | отрицательный пример или диагностика | не делать основным успешным весом\n",
        ),
        md(
            """
            ## Checkpoint 2: генерация конфигурации и реальный запуск обучения

            Ожидаемое время: smoke ACT около 1–3 минут; длительное обучение `ACT_STEPS=5000` обычно занимает от десятков минут до нескольких часов.
            Здесь ACT используется в первую очередь для диагностики: то, что обучение работает, не означает высокой успешности в замкнутом цикле; дальше обязательно нужны строгая оценка и проверка видео.
            """
        ),
        code(
            r'''
DATASET_REPO_ID = os.environ.get("ACT_DATASET_REPO_ID", "datawhale_eai_pnp_language")
TRAIN_DATA_ROOT = Path(os.environ.get("ACT_TRAIN_DATA_ROOT", DATA_ROOT / "demo_data_language"))
ACT_POLICY_PATH = Path(os.environ.get("ACT_POLICY_PATH", MODEL_ROOT / "act_stable61_step2500" / "pretrained_model"))
CONFIG_DIR = OUTPUT_ROOT / "configs"
LOG_DIR = OUTPUT_ROOT / "logs"
RUN_ROOT = OUTPUT_ROOT / "runs" / "act_stable61_repro"
SMOKE_OUTPUT = RUN_ROOT / "smoke"
LONG_OUTPUT = RUN_ROOT / "stable61_full"

smoke_config = make_lerobot_train_config(
    "act", DATASET_REPO_ID, TRAIN_DATA_ROOT, SMOKE_OUTPUT,
    steps=2, batch_size=4, chunk_size=50, n_action_steps=50,
)
long_config = make_lerobot_train_config(
    "act", DATASET_REPO_ID, TRAIN_DATA_ROOT, LONG_OUTPUT,
    steps=int(os.environ.get("ACT_STEPS", "5000")),
    batch_size=int(os.environ.get("ACT_BATCH_SIZE", "8")),
    chunk_size=50,
    n_action_steps=50,
)
smoke_config_path = write_json_yaml(CONFIG_DIR / "act_smoke.yaml", smoke_config)
long_config_path = write_json_yaml(CONFIG_DIR / "act_stable61_full.yaml", long_config)

train_lerobot_config_in_notebook(smoke_config_path, enabled=RUN_SMOKE, progress_name="ACT smoke")
train_lerobot_config_in_notebook(long_config_path, enabled=RUN_LONG_TRAIN, progress_name="ACT long train")
''',
            "Нативная точка входа обучения в Notebook. RUN_SMOKE/RUN_LONG_TRAIN по умолчанию выключены; после обучения обязательно выполните strict closed-loop.\n",
        ),
        md(
            """
            ## Checkpoint 3: просмотр логов обучения и checkpoint в реальном времени

            Ожидаемое время: несколько секунд. Во время длительного обучения эту ячейку можно запускать повторно, чтобы проверить, идёт ли step и сохраняются ли checkpoint.
            """
        ),
        code(
            r'''
print("smoke metrics:")
tail_log(SMOKE_OUTPUT / "notebook_train_metrics.jsonl", lines=20)
print("\nlong train metrics:")
tail_log(LONG_OUTPUT / "notebook_train_metrics.jsonl", lines=40)
print("\ncheckpoints:")
list_checkpoints(LONG_OUTPUT)
''',
        ),
        md(
            """
            ## Checkpoint 4: измеренное сравнение для уже завершённого длительного обучения

            Это ранее выполненное на устройстве AMD сравнение диагностики ACT. Реальный результат длительного обучения в этом Notebook нужно определять по последующей строгой оценке.
            """
        ),
        code(
            r'''
rows = [
    ("stage1 stable42", "8000 steps", "обучается, но замкнутый цикл ещё нестабилен"),
    ("stable61", "5000 steps", "step2500 лучше step5000"),
    ("strict30", "15/30", "текущий защищённый кандидат repair15; три панели 3/10 + 4/10 + 8/10"),
]
md_table(["Этап", "Прогресс обучения", "Вывод"], rows)
''',
            "Этап | Прогресс обучения | Вывод\nstage1 stable42 | 8000 steps | обучается, но замкнутый цикл ещё нестабилен\nstable61 | 5000 steps | step2500 лучше step5000\nstrict30 | 7/30 | текущий защищённый fallback best\n",
        ),
        md(
            """
            ## Checkpoint 5: строгая оценка внутри Notebook

            Ожидаемое время: 30 episode обычно занимают довольно долго; для быстрого порога сначала используйте `ACT_EVAL_EPISODES=6`, затем расширьте до 30.
            Эта ячейка загружает ACT прямо в ядре Notebook и выполняет rollout по каждому seed; внешний скрипт обучения или оценки через shell больше не вызывается.
            """
        ),
        code(
            r'''
eval_episodes = os.environ.get("ACT_EVAL_EPISODES", "30")
eval_seed_start = os.environ.get("ACT_EVAL_SEED_START", "1030")
eval_policy = resolve_eval_policy(ACT_POLICY_PATH, LONG_OUTPUT, "ACT_EVAL_POLICY_PATH")
result_path = OUTPUT_ROOT / f"act_stable61_seed{eval_seed_start}_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "act",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=eval_seed_start,
    render=env_flag("RENDER_EVAL"),
    enabled=RUN_EVAL,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
summarize_jsonl(result_path)
''',
        ),
        md("## Checkpoint 6: почему это диагностический контрпример"),
        code(
            r'''
rows = [
    ("Приближение и контакт", "часто получается", "значит, что визуальные или двигательные способности не отсутствуют полностью"),
    ("Перенос, выравнивание и освобождение", "чаще всего проваливается", "длинные последовательности behavior cloning накапливают ошибку"),
    ("loss", "сам по себе не определяет checkpoint", "нужен strict physical_success и видео"),
    ("Дальнейшее направление", "данные success-only и recovery более высокого качества", "не рекомендуется вслепую наращивать параметры модели"),
]
md_table(["Наблюдение", "Явление", "Как действовать"], rows)
show_image("act_success_sequence.jpg", "Ключевые кадры успеха ACT")
show_image("act_failure_sequence.jpg", "Ключевые кадры провала ACT")
''',
        ),
        md("## Checkpoint 7: вывод для учебника"),
        code(
            r'''
print("""
Линию ACT стоит поместить в диагностический раздел учебника:
1. Цепочка обучения и оценки воспроизводима;
2. Текущий защищённый кандидат repair15 даёт строгие 15/30, историческая цель 17/30 пока не восстановлена полностью;
3. Она хорошо показывает, почему нужны strict physical_success, проверка видео и корзины провалов по этапам.
""")
''',
        ),
    ]


def main() -> None:
    write_nb("14_smolvla_end_to_end.ipynb", smolvla_cells())
    write_nb("15_pi0_end_to_end.ipynb", pi0_cells())
    write_nb("16_act_end_to_end.ipynb", act_cells())


if __name__ == "__main__":
    main()
