#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"
status=0

echo "[1/3] scanning sensitive strings"
if rg -n --hidden \
  --glob '!.git/**' \
  --glob '!scan_public_release.sh' \
  --glob '!*.ipynb' \
  'BEGIN (RSA|OPENSSH|EC|PRIVATE) KEY|HF_TOKEN|AWS_SECRET|password=|token=[A-Za-z0-9_./+=-]{12,}|/[h]ome/[A-Za-z0-9_.-]+|/[d]ata/Data|/run/user/[0-9]+|https://[^ ]+:[^ ]+@' .; then
  echo "sensitive-looking content found"
  status=1
else
  echo "no sensitive-looking content found"
fi

echo "[2/3] scanning oversized files"
large_files="$(find . -type f -size +100M -not -path './.git/*' -print)"
if [[ -n "$large_files" ]]; then
  printf '%s\n' "$large_files"
  echo "files over 100 MiB must be reviewed before release"
  status=1
else
  echo "no files over 100 MiB"
fi

echo "[3/3] checking generated caches"
if find . -type d \( -name .git -o -name __pycache__ -o -name .pytest_cache \) -print -quit | grep -q .; then
  echo "generated cache directories found; remove them from the staged release"
  status=1
else
  echo "no generated cache directories"
fi

exit "$status"
