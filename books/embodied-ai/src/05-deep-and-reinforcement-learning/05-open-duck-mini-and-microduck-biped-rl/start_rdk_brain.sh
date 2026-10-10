#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDK_POLICY_ROOT:-$HOME/microduck_policy_v2}"
PORT="${RDK_POLICY_PORT:-8766}"
cd "$ROOT"

exec python3 -u rdk_multi_policy_server.py \
  --policy walking=models/BEST_alpha_walking.onnx \
  --policy stand=models/BEST_alpha_stand.onnx \
  --policy sitstand=models/BEST_alpha_sitstand.onnx \
  --policy roll=models/roulade.onnx \
  --policy kick_left=models/ball_kick_left.onnx \
  --policy kick_right=models/ball_kick_right.onnx \
  --policy groundpick=models/alpha_ground_pick.onnx \
  --policy drive=models/BEST_roller.onnx \
  --policy crouch=models/BEST_roller_crouch.onnx \
  --host 0.0.0.0 \
  --port "$PORT"
