#!/usr/bin/env bash
# Lower default route MTU inside WSL (common fix when TLS completes but HTTP returns 0 bytes).
# Usage:  bash scripts/wsl_mtu.sh 1350
# Requires: sudo once per boot unless you automate (see config/README.md).

set -euo pipefail
MTU="${1:-1350}"
IFACE="$(ip route show default 2>/dev/null | awk '/default/ {print $5; exit}')"
if [[ -z "${IFACE}" ]]; then
  echo "Could not detect default interface." >&2
  exit 1
fi
echo "Setting MTU=${MTU} on ${IFACE} (needs sudo)..."
sudo ip link set dev "${IFACE}" mtu "${MTU}"
ip link show "${IFACE}" | sed -n '1,2p'
echo "Smoke test:"
curl -4 --http1.1 -sI --max-time 30 'https://api.fda.gov/drug/label.json?limit=1' | head -n 5
