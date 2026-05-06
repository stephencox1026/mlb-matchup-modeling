#!/usr/bin/env bash
# Run in a normal Terminal session (interactive GitHub auth).
# Prerequisite: create an EMPTY repo on GitHub (no README/license), or fix OWNER/REPO below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OWNER="${GITHUB_OWNER:-stephencox}"
REPO="${GITHUB_REPO:-lad-vs-lhp-analysis}"
URL="https://github.com/${OWNER}/${REPO}.git"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$URL"
else
  git remote add origin "$URL"
fi

echo "Pushing to $URL ..."
git push -u origin main
git push origin v0.1.0
echo "Done. Open: https://github.com/${OWNER}/${REPO}"
