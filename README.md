# Conditional GAN for Controllable Shoe Image Generation

This project implements a conditional Generative Adversarial Network (GAN) for generating realistic shoe images with explicit control over high-level attributes such as category, material, closure type, and gender. The model learns the mapping:

```
G(edge_map_of_shoe, attribute_vector) → realistic_shoe_image
```

## Project Structure

```
jisha/
├── dataset.py          # Dataset loader for UT Zappos50K
├── models.py           # Generator, Discriminator, and loss functions
├── train.py            # Training script
├── infer.py            # Inference script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dataset Structure

The UT Zappos50K dataset should be organized as follows:

```
ut-zap50k-images-square/
├── Boots/              # Boot images organized by subcategory
├── Sandals/            # Sandal images organized by subcategory
├── Shoes/              # Shoe images organized by subcategory
├── Slippers/           # Slipper images organized by subcategory
└── ut-zap50k-data/     # Metadata files
    ├── image-path.mat      # MATLAB file with relative image paths
    ├── meta-data-bin.csv   # CSV with CID and binary attribute vectors
    ├── meta-data.csv       # Non-binary metadata
    └── other files...
```

**Key Points:**
- Images are stored in category directories with paths like `Shoes/Oxfords/Bostonian/100627.72.jpg`
- The `.mat` file contains relative paths from the root directory
- The CSV has a CID column followed by binary attribute columns like `Category.Shoes`, `Material.Leather`, etc.
- 50,025 total images with 151 binary attribute columns

## Installation

1. Clone or download the project files
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Note: This code requires PyTorch with CUDA support for optimal performance. Install PyTorch according to your CUDA version from [pytorch.org](https://pytorch.org/).

## Training

### Basic Training Command

```bash
python train.py --data_root ./ut-zap50k-images-square --epochs 200 --batch_size 16
```

### Advanced Training with Custom Parameters

```bash
python train.py \
    --data_root ./ut-zap50k-images-square \
    --epochs 200 \
    --batch_size 16 \
    --image_size 256 \
    --g_lr 0.0002 \
    --d_lr 0.0002 \
    --lambda_l1 100.0 \
    --lambda_perceptual 10.0 \
    --ngf 64 \
    --ndf 64 \
    --checkpoint_dir ./checkpoints \
    --sample_dir ./samples \
    --log_dir ./logs \
    --save_freq 10 \
    --sample_freq 5 \
    --val_freq 1
```

### Filtering Attributes

You can filter which attributes to use during training:

```bash
# Use only Category and Material attributes
python train.py \
    --data_root ./ut-zap50k-images-square \
    --attr_prefixes "Category,Material" \
    --min_attr_freq 50

# Use attributes that appear at least 200 times
python train.py \
    --data_root ./ut-zap50k-images-square \
    --min_attr_freq 200
```

### Resume Training

```bash
python train.py \
    --data_root ./ut-zap50k-images-square \
    --resume_from ./checkpoints/epoch_050.pth
```

## Inference

### Generate Images from Validation Set

```bash
python infer.py \
    --checkpoint ./checkpoints/best.pth \
    --data_root ./ut-zap50k-images-square \
    --output_dir ./inference_results \
    --mode generate \
    --num_samples 64
```

### Attribute Manipulation

Generate images with modified attributes to demonstrate controllable generation:

```bash
python infer.py \
    --checkpoint ./checkpoints/best.pth \
    --data_root ./ut-zap50k-images-square \
    --output_dir ./attribute_manipulation \
    --mode manipulate \
    --sample_idx 0
```

### Attribute Interpolation

Interpolate between two different attribute vectors while keeping the same edge map:

```bash
python infer.py \
    --checkpoint ./checkpoints/best.pth \
    --data_root ./ut-zap50k-images-square \
    --output_dir ./interpolation_results \
    --mode interpolate \
    --sample_idx1 0 \
    --sample_idx2 10 \
    --num_steps 7
```

## Training Arguments

### Data Arguments
- `--data_root`: Root directory of the Zappos dataset (required)
- `--image_size`: Size of input images (default: 256)
- `--train_ratio`: Ratio of data to use for training (default: 0.8)
- `--min_attr_freq`: Minimum frequency for attributes to be included (default: 100)
- `--attr_prefixes`: Comma-separated attribute prefixes to use

### Model Arguments
- `--ngf`: Number of generator filters (default: 64)
- `--ndf`: Number of discriminator filters (default: 64)

### Training Arguments
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 16)
- `--g_lr`: Learning rate for generator (default: 0.0002)
- `--d_lr`: Learning rate for discriminator (default: 0.0002)
- `--beta1`: Beta1 for Adam optimizer (default: 0.5)
- `--beta2`: Beta2 for Adam optimizer (default: 0.999)

### Loss Weights
- `--lambda_l1`: Weight for L1 loss (default: 100.0)
- `--lambda_perceptual`: Weight for perceptual loss (default: 10.0)

### Output Arguments
- `--checkpoint_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--sample_dir`: Directory to save sample images (default: ./samples)
- `--log_dir`: Directory to save logs (default: ./logs)

## Model Architecture

### Generator
- **Architecture**: U-Net with skip connections
- **Input**: Edge maps (3 channels) + attribute vectors
- **Attribute Injection**: At the bottleneck using adaptive normalization
- **Output**: RGB images (3 channels)
- **Activation**: Tanh output, LeakyReLU/ReLU elsewhere

### Discriminator
- **Architecture**: PatchGAN with spectral normalization
- **Input**: Edge maps + attribute spatial maps + images
- **Conditioning**: Concatenation of all inputs
- **Output**: Patch-wise real/fake predictions

### Loss Functions
1. **Hinge GAN Loss**: For adversarial training
2. **L1 Reconstruction Loss**: For pixel-level similarity
3. **Perceptual Loss**: Using VGG16 features for high-level similarity

## Monitoring Training

### Tensorboard
View training progress in real-time:

```bash
tensorboard --logdir ./logs
```

### Sample Images
Generated sample images are saved periodically in `./samples/` during training.

### Checkpoints
- `latest.pth`: Most recent checkpoint
- `best.pth`: Best checkpoint based on validation loss
- `epoch_XXX.pth`: Periodic epoch checkpoints

## Expected Results

The model should learn to:

1. **Generate realistic shoe images** from edge maps and attributes
2. **Control generation** by modifying attribute vectors
3. **Maintain shape consistency** while changing semantic attributes
4. **Interpolate smoothly** between different attribute combinations

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch_size` or `--image_size`
2. **Dataset not found**: Check `--data_root` path and file structure
3. **No attributes found**: Adjust `--min_attr_freq` or `--attr_prefixes`
4. **Training instability**: Adjust learning rates or loss weights

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Increase batch size**: If you have sufficient GPU memory
3. **Monitor loss curves**: Use Tensorboard to track training progress
4. **Experiment with hyperparameters**: Learning rates and loss weights

## File Descriptions

- **dataset.py**: Handles loading of images, attributes, and edge map generation
- **models.py**: Contains Generator, Discriminator, and loss function implementations
- **train.py**: Main training loop with validation and checkpointing
- **infer.py**: Inference script for generation, manipulation, and interpolation

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)
- At least 8GB GPU memory for default settings

## Citation

This implementation is based on the pix2pix architecture with conditional extensions for attribute control.