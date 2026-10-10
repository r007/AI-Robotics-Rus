# Материалы главы AGILE

Этот каталог хранит только лёгкие учебные материалы, нужные для главы. Здесь нет весов моделей AGILE, окружений Isaac, кэша Nucleus и полных журналов запусков.

## official_figures

Следующие изображения взяты из каталога `docs/figures/` официального открытого репозитория NVIDIA Isaac `nvidia-isaac/WBC-AGILE`:

- `agile_highlights.png`
- `separate_upper_lower_body_policy_diagram.png`
- `evaluation_report_summary.png`
- `evaluation_report_tracking.png`

При использовании в тексте сохраняйте подпись к рисунку и указание источника.

## official_videos

Следующие GIF взяты из каталога `docs/videos/` официального открытого репозитория `nvidia-isaac/WBC-AGILE`:

- `booster_t1_vel_sim2sim.gif`
- `booster_t1_vel_sim2real.gif`
- `unitree_g1_vel_height_sim2sim.gif`
- `unitree_g1_vel_height_sim2real.gif`
- `g1_apple_grasp_black_sort_bin_multi_objects_no_marker_reduced.gif`
- `unitree_g1_dancing_sim.gif`

Эти материалы показывают результаты официальной статьи и проекта; они не являются результатом повторного обучения, выполненного в этой главе.

## local_videos

Следующие MP4 созданы в ходе воспроизведения в этой главе:

- `t1_velocity_policy_smoke.mp4`: короткое видео для предобученной политики `Velocity-T1-v0`.
- `g1_velocity_height_checkpoint_smoke.mp4`: короткое видео для полного чекпоинта `Velocity-Height-G1-Distillation-Recurrent-v0`.
- `g1_pickplace_tracking_scene_smoke.mp4`: короткое видео загрузки сцены `G1-PickPlace-Tracking-v0` и цепочки действий.

Эти видео — подтверждение работоспособности (smoke-test). Они показывают только то, что окружение, ассеты, загрузка политики или отладочная цепочка сцены работают; они не доказывают завершённого обучения и не означают развёртывания на реальном роботе.
