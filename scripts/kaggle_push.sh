#!/bin/bash
# ==============================================================
#  MedBharat-1: Push to Kaggle & Trigger Training
#
#  Usage:
#    ./scripts/kaggle_push.sh                  # full: data + kernel
#    ./scripts/kaggle_push.sh --data-only      # upload data only
#    ./scripts/kaggle_push.sh --kernel-only    # push kernel only
#
#  Prerequisites:
#    pip install kaggle
#    ~/.kaggle/kaggle.json must exist
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data"
NOTEBOOK_PATH="${PROJECT_DIR}/kaggle_notebook.ipynb"

# Parse args
DATA_ONLY=false
KERNEL_ONLY=false
for arg in "$@"; do
    case $arg in
        --data-only)   DATA_ONLY=true ;;
        --kernel-only) KERNEL_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--data-only] [--kernel-only]"
            echo "  --data-only    Upload dataset only"
            echo "  --kernel-only  Push notebook kernel only"
            exit 0 ;;
    esac
done

echo "============================================"
echo "  MedBharat-1: Kaggle Push & Train"
echo "============================================"

# --- Validate prerequisites ---
if ! command -v kaggle &>/dev/null; then
    echo "ERROR: kaggle CLI not found."
    echo "  Install: pip install kaggle"
    exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found"
    echo "  Get your token: https://www.kaggle.com/settings -> API -> Create New Token"
    exit 1
fi

if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "ERROR: Notebook not found at $NOTEBOOK_PATH"
    exit 1
fi

# Get Kaggle username
KAGGLE_USERNAME=$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
DATASET_SLUG="${KAGGLE_USERNAME}/medbharat-healthcare-data"
KERNEL_SLUG="${KAGGLE_USERNAME}/medbharat-1-training"

echo "  Username: ${KAGGLE_USERNAME}"
echo "  Dataset:  ${DATASET_SLUG}"
echo "  Kernel:   ${KERNEL_SLUG}"
echo ""

# --- Step 1: Upload dataset ---
if [ "$KERNEL_ONLY" = false ]; then
    echo "[1/3] Uploading dataset to Kaggle..."

    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "  ERROR: Data directory empty: $DATA_DIR"
        echo "  Run first: python scripts/prepare_data.py --download"
        exit 1
    fi

    # Flatten data into a temp upload directory (Kaggle needs flat files)
    UPLOAD_DIR=$(mktemp -d)
    echo "  Flattening data for upload..."
    python3 -c "
import shutil, os
from pathlib import Path
src = Path('${DATA_DIR}')
dst = Path('${UPLOAD_DIR}')
count = 0
for f in src.rglob('*'):
    if f.is_file() and f.suffix in ('.gz', '.jsonl', '.txt'):
        # Prefix with parent dir name to avoid collisions
        parent = f.parent.name
        out_name = f'{parent}__{f.name}' if parent != src.name else f.name
        shutil.copy2(f, dst / out_name)
        count += 1
print(f'  Copied {count} files to upload directory')
"

    DATA_SIZE=$(du -sh "$UPLOAD_DIR" 2>/dev/null | cut -f1)
    N_FILES=$(ls "$UPLOAD_DIR" | wc -l | tr -d ' ')
    echo "  Upload: ${N_FILES} files, ${DATA_SIZE}"

    # Create dataset-metadata.json
    python3 -c "
import json, os
meta = {
    'title': 'MedBharat Healthcare Data',
    'id': '${DATASET_SLUG}',
    'licenses': [{'name': 'CC0-1.0'}],
    'keywords': ['healthcare', 'medical', 'india', 'nlp']
}
with open(os.path.join('${UPLOAD_DIR}', 'dataset-metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)
"

    # Check if dataset exists, create or update
    if kaggle datasets list --mine 2>/dev/null | grep -q "medbharat-healthcare-data"; then
        echo "  Dataset exists, creating new version..."
        kaggle datasets version -p "$UPLOAD_DIR" -m "Update $(date +%Y-%m-%d-%H%M)"
    else
        echo "  Creating new dataset..."
        kaggle datasets create -p "$UPLOAD_DIR"
    fi

    rm -rf "$UPLOAD_DIR"

    echo "  Dataset uploaded: ${DATASET_SLUG}"

    if [ "$DATA_ONLY" = true ]; then
        echo ""
        echo "Done! Dataset uploaded."
        echo "View: https://www.kaggle.com/datasets/${DATASET_SLUG}"
        exit 0
    fi

    echo "  Waiting 60s for Kaggle to process dataset..."
    sleep 60
fi

# --- Step 2: Push notebook as kernel ---
echo "[2/3] Pushing notebook as Kaggle kernel..."

KERNEL_META_DIR=$(mktemp -d)
cp "$NOTEBOOK_PATH" "$KERNEL_META_DIR/"

python3 -c "
import json
meta = {
    'id': '${KERNEL_SLUG}',
    'title': 'MedBharat-1 Training',
    'code_file': 'kaggle_notebook.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'gpu_type': 'T4',
    'enable_internet': True,
    'dataset_sources': ['${DATASET_SLUG}'],
    'competition_sources': [],
    'kernel_sources': [],
    'docker_image_pinning_type': 'latest'
}
with open('${KERNEL_META_DIR}/kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
"

kaggle kernels push -p "$KERNEL_META_DIR"
rm -rf "$KERNEL_META_DIR"

echo "  Kernel pushed: ${KERNEL_SLUG}"

# --- Step 3: Done ---
echo ""
echo "[3/3] Training triggered!"
echo ""
echo "============================================"
echo "  MedBharat-1 Training Running on Kaggle"
echo "============================================"
echo ""
echo "  Dataset: https://www.kaggle.com/datasets/${DATASET_SLUG}"
echo "  Kernel:  https://www.kaggle.com/code/${KERNEL_SLUG}"
echo ""
echo "  Monitor:"
echo "    kaggle kernels status ${KERNEL_SLUG}"
echo ""
echo "  View logs:"
echo "    kaggle kernels log ${KERNEL_SLUG}"
echo ""
echo "  Download output (after completion):"
echo "    kaggle kernels output ${KERNEL_SLUG} -p ./kaggle_output"
echo ""
echo "============================================"
