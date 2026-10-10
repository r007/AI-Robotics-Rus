# Краткая справка по командам LeHome

## Загрузка ассетов

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

## Загрузка объединённого обучающего набора

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

## Обучение ACT

```bash
mkdir -p /path/to/outputs/train/act_top_short

lerobot-train \
  --config_path 15-challenges/LeHome/resources/configs/train_act_every_embodied.yaml \
  2>&1 | tee /path/to/outputs/train/act_top_short/train.log
```

## Обучение DP

```bash
mkdir -p /path/to/outputs/train/dp_top_short

lerobot-train \
  --config_path 15-challenges/LeHome/resources/configs/train_dp_every_embodied.yaml \
  2>&1 | tee /path/to/outputs/train/dp_top_short/train.log
```

## Оценка ACT

```bash
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path /path/to/outputs/train/act_top_short/checkpoints/last/pretrained_model \
  --dataset_root Datasets/example/top_short_merged \
  --garment_type top_short \
  --num_episodes 2 \
  --enable_cameras \
  --save_video \
  --video_dir /path/to/outputs/eval/act_top_short \
  --device cpu
```

## Оценка DP

```bash
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path /path/to/outputs/train/dp_top_short/checkpoints/last/pretrained_model \
  --dataset_root Datasets/example/top_short_merged \
  --garment_type top_short \
  --num_episodes 2 \
  --enable_cameras \
  --save_video \
  --video_dir /path/to/outputs/eval/dp_top_short \
  --device cpu \
  --policy_device cpu \
  --policy_num_inference_steps 1
```

## Разбор журнала обучения и построение графиков

```bash
python 15-challenges/LeHome/resources/scripts/plot_train_metrics.py \
  --log_file /path/to/outputs/train/act_top_short/train.log \
  --out_dir /path/to/outputs/plots/act_top_short \
  --title "ACT Top-Short Training Metrics"
```

## Длительное обучение ACT для соревнования

```bash
cd /path/to/lehome-challenge
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --config_path 15-challenges/LeHome/resources/configs/train_act_competition_l40.yaml \
  2>&1 | tee /path/to/outputs/train/act_four_types_l40.log
```

## Выборка метрик GPU

```bash
15-challenges/LeHome/resources/scripts/sample_gpu_metrics.sh \
  /path/to/outputs/monitor/act_four_types_l40_gpu.csv \
  10
```

## Построение графика GPU

```bash
python 15-challenges/LeHome/resources/scripts/plot_gpu_metrics.py \
  --csv_file /path/to/outputs/monitor/act_four_types_l40_gpu.csv \
  --out_png /path/to/outputs/plots/act_four_types_l40_live/gpu_metrics.png \
  --title "L40 GPU Monitor During ACT Training"
```
