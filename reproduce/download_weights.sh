#!/bin/bash
# Download RNA-FM pretrained weights
# Source: cuhkaih/rnafm on HuggingFace
set -e

CACHE_DIR="${HOME}/.cache/torch/hub/checkpoints"
mkdir -p "${CACHE_DIR}"

echo "Downloading RNA-FM pretrained weights from HuggingFace..."
curl -L -o "${CACHE_DIR}/RNA-FM_pretrained.pth" \
  "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"

echo "Done! Downloaded to ${CACHE_DIR}/RNA-FM_pretrained.pth"
ls -lh "${CACHE_DIR}/RNA-FM_pretrained.pth"
