#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENET_DIR="${1:-${REPO_ROOT}/external/PARENet}"
ADAPTER_DIR="${REPO_ROOT}/parenet_protein/ProteinFit"
DEST_DIR="${PARENET_DIR}/experiments/ProteinFit"

if [[ ! -d "${PARENET_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${PARENET_DIR}")"
  git clone https://github.com/yaorz97/PARENet.git "${PARENET_DIR}"
fi

mkdir -p "${DEST_DIR}"
cp "${PARENET_DIR}/experiments/3DMatch/model.py" "${DEST_DIR}/model.py"
cp "${PARENET_DIR}/experiments/3DMatch/backbone.py" "${DEST_DIR}/backbone.py"
cp "${PARENET_DIR}/experiments/3DMatch/trainval.py" "${DEST_DIR}/trainval.py"
cp "${ADAPTER_DIR}/config.py" "${DEST_DIR}/config.py"
cp "${ADAPTER_DIR}/dataset.py" "${DEST_DIR}/dataset.py"
cp "${ADAPTER_DIR}/loss.py" "${DEST_DIR}/loss.py"
cp "${ADAPTER_DIR}/inspect_dataset.py" "${DEST_DIR}/inspect_dataset.py"
cp "${ADAPTER_DIR}/audit_training_windows.py" "${DEST_DIR}/audit_training_windows.py"
cp "${ADAPTER_DIR}/smoke_train.py" "${DEST_DIR}/smoke_train.py"
cp "${ADAPTER_DIR}/test.py" "${DEST_DIR}/test.py"

echo "Installed ProteinFit experiment at: ${DEST_DIR}"
echo "Dataset default: ${REPO_ROOT}/dataset"
echo "Next: cd ${PARENET_DIR} && follow upstream installation instructions."
