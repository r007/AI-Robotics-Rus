#!/usr/bin/env bash
set -euo pipefail

# Fetch public references outside the tutorial repository so they are not
# accidentally vendored into the Datawhale release.
DEST="${1:-${TMPDIR:-$HOME/.cache/atec2026-reference}}"
mkdir -p "$DEST"

clone_or_update() {
  local url="$1"
  local name="$2"
  if [[ -d "$DEST/$name/.git" ]]; then
    git -C "$DEST/$name" fetch --depth=1 origin main
    git -C "$DEST/$name" reset --hard FETCH_HEAD
  else
    git clone --depth=1 "$url" "$DEST/$name"
  fi
}

clone_or_update \
  "https://github.com/cicaburnwood-crypto/SLAM_ATEC2026_Simulation_Challenge.git" \
  "SLAM_ATEC2026_Simulation_Challenge"
clone_or_update \
  "https://github.com/StevenLiudw/Clear_ATEC2026_Simulation_Challenge.git" \
  "Clear_ATEC2026_Simulation_Challenge"

printf 'references available under %s\n' "$DEST"
