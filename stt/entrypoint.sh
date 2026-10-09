#!/bin/bash
set -e

# Prioritize host NVIDIA drivers mounted by the NVIDIA Container Toolkit.
# On systems with modern NVIDIA drivers (>=545), having cuda-compat (/usr/local/cuda-12.3/compat)
# in LD_LIBRARY_PATH causes CUDA initialization to fail with:
# "failed to initialize CUDA: system has unsupported display driver / cuda driver combination".
# This script ensures native host libraries are used on modern systems while preserving
# backward compatibility for older legacy drivers.

CLEAN_LD_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'compat' | paste -sd:)

if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | cut -d'.' -f1 | head -n1 || echo "")
fi

if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -lt 525 ] 2>/dev/null; then
    # Legacy host driver needing forward-compatibility package
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
else
    # Modern driver (>=525) or standard container runtime: use native host libraries
    export LD_LIBRARY_PATH="${CLEAN_LD_PATH:-/usr/local/nvidia/lib:/usr/local/nvidia/lib64}"
fi

exec /app/whisper-server "$@"
