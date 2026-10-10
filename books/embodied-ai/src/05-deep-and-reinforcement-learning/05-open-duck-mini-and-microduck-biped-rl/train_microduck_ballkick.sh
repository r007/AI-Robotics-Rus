#!/usr/bin/env bash
set -euo pipefail

MICRODUCK_RL="${MICRODUCK_RL:-$PWD}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-6000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
WANDB_MODE="${WANDB_MODE:-offline}"
UV_BIN="${UV_BIN:-uv}"

cd "$MICRODUCK_RL"
export WANDB_MODE

echo "repo=$MICRODUCK_RL"
echo "task=Mjlab-BallKick-Flat-MicroDuck"
echo "num_envs=$NUM_ENVS max_iterations=$MAX_ITERATIONS save_interval=$SAVE_INTERVAL"
echo "wandb_mode=$WANDB_MODE"

"$UV_BIN" run train Mjlab-BallKick-Flat-MicroDuck \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.save-interval "$SAVE_INTERVAL"
