#!/bin/bash
# ============================================================
# Lightning AI Setup Script for Pix2Pix GAN Training
# ============================================================
#
# Run this inside your Lightning AI Studio terminal:
#   bash lightning_setup.sh
#
# BEFORE RUNNING: Upload ut-zap50k-images-square.zip to the Studio
# ============================================================

set -e

echo "=========================================="
echo "  Pix2Pix GAN - Lightning AI Setup"
echo "=========================================="

# Step 1: Install dependencies
echo ""
echo "[1/3] Installing Python dependencies..."
pip install -q torch torchvision numpy opencv-python scipy pandas matplotlib tqdm Pillow tensorboard

# Step 2: Check for dataset
echo ""
echo "[2/3] Checking dataset..."
if [ -d "ut-zap50k-images-square" ]; then
    echo "  ✅ Dataset found!"
elif [ -f "ut-zap50k-images-square.zip" ]; then
    echo "  Found zip file, extracting..."
    unzip -q ut-zap50k-images-square.zip
    echo "  ✅ Dataset extracted!"
else
    echo "  ⚠️  Dataset NOT found!"
    echo "  Please upload ut-zap50k-images-square.zip to this directory."
    echo ""
    echo "  Expected structure:"
    echo "    ut-zap50k-images-square/"
    echo "      ├── Boots/"
    echo "      ├── Sandals/"
    echo "      ├── Shoes/"
    echo "      ├── Slippers/"
    echo "      └── ut-zap50k-data/"
    echo "          ├── image-path.mat"
    echo "          └── meta-data-bin.csv"
    exit 1
fi

# Step 3: Verify GPU
echo ""
echo "[3/3] Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  ✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ✅ VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  ⚠️  No GPU detected! Training will be slow.')
"

echo ""
echo "=========================================="
echo "  ✅ Setup complete! Start training with:"
echo "=========================================="
echo ""
echo "  python train.py --data_root ./ut-zap50k-images-square \\"
echo "    --epochs 30 --batch_size 32 --num_workers 4 --image_size 256"
echo ""
echo "  Estimated: ~3-5 hours on T4 GPU"
echo ""
