#!/usr/bin/env bash
# Remove large data paths from ALL commits so GitHub accepts `git push`.
# Run from repo root: bash scripts/strip_data_from_git_history.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "Install git-filter-repo first, e.g.:"
  echo "  pip install git-filter-repo"
  echo "  # or: sudo apt install git-filter-repo"
  exit 1
fi

# Paths that triggered GH001 / size limits (adjust if git log shows others).
git filter-repo --force --invert-paths \
  --path data/raw/ \
  --path data/processed/openfda.pkl

echo ""
echo "History rewritten. Restore remote if filter-repo removed it:"
echo "  git remote add origin git@github.com:shreyakumari0301/Sim.git"
echo "Then force-push (rewrites remote history):"
echo "  git push --force-with-lease origin main"
