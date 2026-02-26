import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms

from dataset import ZapposDataset, create_dataloader
from models import Generator, PatchDiscriminator


class ConditionalGANInference:
    """Inference class for conditional GAN"""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")

        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.args = self.checkpoint['args']
        self.attr_dim = self.checkpoint['attr_dim']

        print(f"Loaded checkpoint from epoch {self.checkpoint['epoch']}")
        print(f"Attribute dimension: {self.attr_dim}")

        # Initialize models
        self._setup_models()

    def _setup_models(self):
        """Setup and load trained models"""
        # Initialize generator
        self.generator = Generator(
            attr_dim=self.attr_dim,
            ngf=self.args.ngf
        ).to(self.device)

        # Load trained weights
        self.generator.load_state_dict(self.checkpoint['generator_state_dict'])
        self.generator.eval()

        print("Models loaded successfully")

    def generate_images(
        self,
        edges: torch.Tensor,
        attributes: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate images from edge maps and attributes

        Args:
            edges: Edge maps [B, 3, H, W]
            attributes: Attribute vectors [B, attr_dim]

        Returns:
            generated_images: [B, 3, H, W]
        """
        with torch.no_grad():
            edges = edges.to(self.device)
            attributes = attributes.to(self.device)

            generated = self.generator(edges, attributes)

        return generated

    def interpolate_attributes(
        self,
        edges: torch.Tensor,
        attr1: torch.Tensor,
        attr2: torch.Tensor,
        num_steps: int = 5
    ) -> torch.Tensor:
        """
        Generate images with interpolated attributes

        Args:
            edges: Edge maps [B, 3, H, W]
            attr1: First attribute vector [B, attr_dim]
            attr2: Second attribute vector [B, attr_dim]
            num_steps: Number of interpolation steps

        Returns:
            interpolated_images: [B*num_steps, 3, H, W]
        """
        with torch.no_grad():
            edges = edges.to(self.device)
            attr1 = attr1.to(self.device)
            attr2 = attr2.to(self.device)

            # Create interpolation weights
            alphas = torch.linspace(0, 1, num_steps, device=self.device)

            all_images = []
            for alpha in alphas:
                # Interpolate attributes
                interp_attrs = (1 - alpha) * attr1 + alpha * attr2

                # Generate images
                generated = self.generator(edges, interp_attrs)
                all_images.append(generated)

            # Concatenate all images
            interpolated_images = torch.cat(all_images, dim=0)

        return interpolated_images

    def modify_attributes(
        self,
        edges: torch.Tensor,
        base_attributes: torch.Tensor,
        attribute_modifications: List[dict]
    ) -> torch.Tensor:
        """
        Generate images with modified attributes

        Args:
            edges: Edge maps [B, 3, H, W]
            base_attributes: Base attribute vectors [B, attr_dim]
            attribute_modifications: List of {column_idx: value} modifications

        Returns:
            modified_images: [B*len(modifications), 3, H, W]
        """
        with torch.no_grad():
            edges = edges.to(self.device)
            base_attributes = base_attributes.to(self.device)

            all_images = []
            for modifications in attribute_modifications:
                # Start with base attributes
                modified_attrs = base_attributes.clone()

                # Apply modifications
                for col_idx, value in modifications.items():
                    if col_idx < self.attr_dim:
                        modified_attrs[:, col_idx] = value

                # Generate images
                generated = self.generator(edges, modified_attrs)
                all_images.append(generated)

            # Concatenate all images
            modified_images = torch.cat(all_images, dim=0)

        return modified_images


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
    title: Optional[str] = None
):
    """Save images as a grid"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if normalize:
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2

    vutils.save_image(images, save_path, nrow=nrow, normalize=False)

    if title:
        print(f"Saved {title} to {save_path}")


def create_comparison_grid(
    edges: torch.Tensor,
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    save_path: str,
    max_samples: int = 8
):
    """Create and save comparison grid (edge, real, generated)"""
    num_samples = min(edges.shape[0], max_samples)

    # Take subset
    edges = edges[:num_samples]
    real_images = real_images[:num_samples]
    generated_images = generated_images[:num_samples]

    # Normalize edge maps for visualization
    edges_vis = (edges + 1) / 2

    # Normalize images
    real_vis = (real_images + 1) / 2
    gen_vis = (generated_images + 1) / 2

    # Create grid: [edges, real, generated] for each sample
    comparison = torch.cat([edges_vis, real_vis, gen_vis], dim=0)

    # Save grid
    vutils.save_image(
        comparison,
        save_path,
        nrow=num_samples,
        normalize=False
    )

    print(f"Saved comparison grid to {save_path}")


def run_inference(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    num_samples: int = 64,
    batch_size: int = 16
):
    """Run inference on validation set"""
    print("Starting inference...")

    # Initialize inference class
    inferencer = ConditionalGANInference(checkpoint_path)

    attr_cols = inferencer.checkpoint["dataset_meta"]["attribute_columns"]

    val_dataset = ZapposDataset(
        data_root=data_root,
        image_size=inferencer.args.image_size,
        split='val',
        train_ratio=inferencer.args.train_ratio,
        min_attribute_freq=inferencer.args.min_attr_freq,
        attribute_prefixes=inferencer.args.attr_prefixes.split(',') if inferencer.args.attr_prefixes else None,
        attribute_columns=attr_cols,
    )

    # Load validation dataset
    val_dataset = ZapposDataset(
        data_root=data_root,
        image_size=inferencer.args.image_size,
        split='val',
        train_ratio=inferencer.args.train_ratio,
        min_attribute_freq=inferencer.args.min_attr_freq,
        attribute_prefixes=inferencer.args.attr_prefixes.split(',') if inferencer.args.attr_prefixes else None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate samples
    all_edges = []
    all_real = []
    all_generated = []
    samples_collected = 0

    print(f"Generating {num_samples} samples...")

    with torch.no_grad():
        for batch_idx, (edges, attributes, real_images) in enumerate(tqdm(val_loader)):
            if samples_collected >= num_samples:
                break

            # Generate fake images
            fake_images = inferencer.generate_images(edges, attributes)

            # Store samples
            batch_size_actual = edges.shape[0]
            remaining = min(batch_size_actual, num_samples - samples_collected)

            all_edges.append(edges[:remaining].cpu())
            all_real.append(real_images[:remaining].cpu())
            all_generated.append(fake_images[:remaining].cpu())

            samples_collected += remaining

    # Concatenate all samples
    all_edges = torch.cat(all_edges, dim=0)
    all_real = torch.cat(all_real, dim=0)
    all_generated = torch.cat(all_generated, dim=0)

    print(f"Generated {all_generated.shape[0]} samples")

    # Save individual grids
    save_image_grid(
        all_edges,
        os.path.join(output_dir, 'edge_maps.png'),
        nrow=8,
        title="Edge maps"
    )

    save_image_grid(
        all_real,
        os.path.join(output_dir, 'real_images.png'),
        nrow=8,
        title="Real images"
    )

    save_image_grid(
        all_generated,
        os.path.join(output_dir, 'generated_images.png'),
        nrow=8,
        title="Generated images"
    )

    # Save comparison grid
    create_comparison_grid(
        all_edges,
        all_real,
        all_generated,
        os.path.join(output_dir, 'comparison.png'),
        max_samples=8
    )


def run_attribute_manipulation(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    sample_idx: int = 0
):
    """Run attribute manipulation experiment"""
    print("Running attribute manipulation...")

    # Initialize inference class
    inferencer = ConditionalGANInference(checkpoint_path)

    attr_cols = inferencer.checkpoint["dataset_meta"]["attribute_columns"]

    val_dataset = ZapposDataset(
        data_root=data_root,
        image_size=inferencer.args.image_size,
        split='val',
        train_ratio=inferencer.args.train_ratio,
        min_attribute_freq=inferencer.args.min_attr_freq,
        attribute_prefixes =inferencer.args.attr_prefixes.split(',') if inferencer.args.attr_prefixes else None,
        attribute_columns=attr_cols,
    )

    # Load validation dataset
    val_dataset = ZapposDataset(
        data_root=data_root,
        image_size=inferencer.args.image_size,
        split='val',
        train_ratio=inferencer.args.train_ratio,
        min_attribute_freq=inferencer.args.min_attr_freq,
        attribute_prefixes=inferencer.args.attr_prefixes.split(',') if inferencer.args.attr_prefixes else None
    )

    # Get a sample
    edges, attributes, real_image = val_dataset[sample_idx]

    # Add batch dimension
    edges = edges.unsqueeze(0)
    attributes = attributes.unsqueeze(0)
    real_image = real_image.unsqueeze(0)

    # Create attribute modifications
    modifications = [
        {},  # Original
    ]

    # Try flipping some attributes
    for i in range(min(5, inferencer.attr_dim)):
        if attributes[0, i] > 0:
            # Turn off this attribute
            modifications.append({i: 0.0})
        else:
            # Turn on this attribute
            modifications.append({i: 1.0})

    # Generate images with modified attributes
    modified_images = inferencer.modify_attributes(edges, attributes, modifications)

    # Create grid with original image
    all_images = torch.cat([real_image, modified_images], dim=0)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_image_grid(
        all_images,
        os.path.join(output_dir, f'attribute_manipulation_sample_{sample_idx}.png'),
        nrow=len(modifications) + 1,
        title=f"Attribute manipulation for sample {sample_idx}"
    )


def run_attribute_interpolation(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    sample_idx1: int = 0,
    sample_idx2: int = 1,
    num_steps: int = 7
):
    """Run attribute interpolation experiment"""
    print("Running attribute interpolation...")

    # Initialize inference class
    inferencer = ConditionalGANInference(checkpoint_path)

    attr_cols = inferencer.checkpoint["dataset_meta"]["attribute_columns"]

    val_dataset = ZapposDataset(
        data_root=data_root,
        image_size=inferencer.args.image_size,
        split='val',
        train_ratio=inferencer.args.train_ratio,
        min_attribute_freq=inferencer.args.min_attr_freq,
        attribute_prefixes=inferencer.args.attr_prefixes.split(',') if inferencer.args.attr_prefixes else None,
        attribute_columns=attr_cols,
    )

    # Load validation dataset
    val_dataset = ZapposDataset(
        data_root=data_root,
        image_size=inferencer.args.image_size,
        split='val',
        train_ratio=inferencer.args.train_ratio,
        min_attribute_freq=inferencer.args.min_attr_freq,
        attribute_prefixes=inferencer.args.attr_prefixes.split(',') if inferencer.args.attr_prefixes else None
    )

    # Get two samples
    edges1, attr1, real1 = val_dataset[sample_idx1]
    edges2, attr2, real2 = val_dataset[sample_idx2]

    # Use same edge map, interpolate attributes
    edges = edges1.unsqueeze(0)
    attr1 = attr1.unsqueeze(0)
    attr2 = attr2.unsqueeze(0)

    # Generate interpolated images
    interpolated = inferencer.interpolate_attributes(edges, attr1, attr2, num_steps)

    # Create grid
    all_images = torch.cat([real1.unsqueeze(0), interpolated, real2.unsqueeze(0)], dim=0)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_image_grid(
        all_images,
        os.path.join(output_dir, f'attribute_interpolation_{sample_idx1}_{sample_idx2}.png'),
        nrow=num_steps + 2,
        title=f"Attribute interpolation between samples {sample_idx1} and {sample_idx2}"
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Inference for Conditional GAN')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the Zappos dataset')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save inference results')

    # Inference options
    parser.add_argument('--mode', type=str, default='generate',
                       choices=['generate', 'manipulate', 'interpolate'],
                       help='Inference mode')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')

    # For manipulation and interpolation
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index for manipulation')
    parser.add_argument('--sample_idx1', type=int, default=0,
                       help='First sample index for interpolation')
    parser.add_argument('--sample_idx2', type=int, default=1,
                       help='Second sample index for interpolation')
    parser.add_argument('--num_steps', type=int, default=7,
                       help='Number of interpolation steps')

    parser.add_argument('--device', type=str, default=None,
                       help='Device to run inference on')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'generate':
        run_inference(
            args.checkpoint,
            args.data_root,
            args.output_dir,
            args.num_samples,
            args.batch_size
        )
    elif args.mode == 'manipulate':
        run_attribute_manipulation(
            args.checkpoint,
            args.data_root,
            args.output_dir,
            args.sample_idx
        )
    elif args.mode == 'interpolate':
        run_attribute_interpolation(
            args.checkpoint,
            args.data_root,
            args.output_dir,
            args.sample_idx1,
            args.sample_idx2,
            args.num_steps
        )

    print("Inference completed!")