import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from dataset import create_dataloader
from models import (
    Generator, PatchDiscriminator, PerceptualLoss,
    hinge_loss_generator, hinge_loss_discriminator
)


class Trainer:
    """Training class for conditional GAN"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create directories
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        # Initialize tensorboard
        self.writer = SummaryWriter(args.log_dir)

        # Setup data
        self._setup_data()

        # Setup models
        self._setup_models()

        # Setup optimizers
        self._setup_optimizers()

        # Setup loss functions
        self._setup_losses()

        # AMP (Automatic Mixed Precision) for faster training on GPU
        self.use_amp = self.device.type == 'cuda'
        self.g_scaler = GradScaler(enabled=self.use_amp)
        self.d_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("Using AMP (mixed precision) for faster training")

        # Save dataset metadata for reproducibility
        train_ds = self.train_loader.dataset
        self.dataset_meta = {
            'attribute_columns': train_ds.attribute_columns,
            'attr_dim': train_ds.attr_dim,
            'indices_train': train_ds.indices.tolist(),
        }

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # Load checkpoint if resuming
        if args.resume_from:
            self._load_checkpoint(args.resume_from)

    def _setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")

        # Get first dataset to determine attribute dimension
        from dataset import ZapposDataset
        temp_dataset = ZapposDataset(
            data_root=self.args.data_root,
            image_size=self.args.image_size,
            split='train',
            min_attribute_freq=self.args.min_attr_freq,
            attribute_prefixes=self.args.attr_prefixes.split(',') if self.args.attr_prefixes else None
        )
        self.attr_dim = temp_dataset.attr_dim
        print(f"Attribute dimension: {self.attr_dim}")

        # Create data loaders
        self.train_loader, self.val_loader = create_dataloader(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            image_size=self.args.image_size,
            train_ratio=self.args.train_ratio,
            min_attribute_freq=self.args.min_attr_freq,
            attribute_prefixes=self.args.attr_prefixes.split(',') if self.args.attr_prefixes else None
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

    def _setup_models(self):
        """Setup generator and discriminator"""
        print("Setting up models...")

        self.generator = Generator(
            attr_dim=self.attr_dim,
            ngf=self.args.ngf
        ).to(self.device)

        self.discriminator = PatchDiscriminator(
            attr_dim=self.attr_dim,
            ndf=self.args.ndf
        ).to(self.device)

        # Print model info
        g_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")

    def _setup_optimizers(self):
        """Setup optimizers"""
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.args.g_lr,
            betas=(self.args.beta1, self.args.beta2)
        )

        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.args.d_lr,
            betas=(self.args.beta1, self.args.beta2)
        )

        # Learning rate schedulers
        self.g_scheduler = optim.lr_scheduler.MultiStepLR(
            self.g_optimizer,
            milestones=[self.args.epochs // 2, self.args.epochs * 3 // 4],
            gamma=0.5
        )

        self.d_scheduler = optim.lr_scheduler.MultiStepLR(
            self.d_optimizer,
            milestones=[self.args.epochs // 2, self.args.epochs * 3 // 4],
            gamma=0.5
        )

    def _setup_losses(self):
        """Setup loss functions"""
        self.perceptual_loss = PerceptualLoss(device=self.device)
        self.l1_loss = nn.L1Loss()

    def _train_discriminator(self, edges, attributes, real_images, fake_images):
        """Train discriminator for one step"""
        self.d_optimizer.zero_grad()

        with autocast(enabled=self.use_amp):
            # Real samples
            real_output = self.discriminator(edges, attributes, real_images)

            # Fake samples (detached to avoid backprop through generator)
            fake_output = self.discriminator(edges, attributes, fake_images.detach())

            # Hinge loss
            d_loss = hinge_loss_discriminator(real_output, fake_output)

        self.d_scaler.scale(d_loss).backward()
        self.d_scaler.step(self.d_optimizer)
        self.d_scaler.update()

        return {
            'd_loss': d_loss.item(),
            'real_score': real_output.mean().item(),
            'fake_score': fake_output.mean().item()
        }

    def _train_generator(self, edges, attributes, real_images):
        """Train generator for one step"""
        self.g_optimizer.zero_grad()

        with autocast(enabled=self.use_amp):
            # Generate fake images
            fake_images = self.generator(edges, attributes)

            # Discriminator output for fake images
            fake_output = self.discriminator(edges, attributes, fake_images)

            # Hinge loss for generator
            g_adv_loss = hinge_loss_generator(fake_output)

            # L1 reconstruction loss
            g_l1_loss = self.l1_loss(fake_images, real_images)

            # Perceptual loss
            g_perc_loss = self.perceptual_loss(fake_images, real_images)

            # Combined generator loss
            g_loss = (g_adv_loss +
                     self.args.lambda_l1 * g_l1_loss +
                     self.args.lambda_perceptual * g_perc_loss)

        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.step(self.g_optimizer)
        self.g_scaler.update()

        return {
            'g_loss': g_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_l1_loss': g_l1_loss.item(),
            'g_perc_loss': g_perc_loss.item()
        }, fake_images.float()

    def _validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.generator.eval()
        self.discriminator.eval()

        total_g_loss = 0
        total_d_loss = 0
        total_l1_loss = 0
        num_batches = 0

        with torch.no_grad():
            for edges, attributes, real_images in self.val_loader:
                edges = edges.to(self.device)
                attributes = attributes.to(self.device)
                real_images = real_images.to(self.device)

                # Generate fake images
                fake_images = self.generator(edges, attributes)

                # Discriminator losses
                real_output = self.discriminator(edges, attributes, real_images)
                fake_output = self.discriminator(edges, attributes, fake_images)
                d_loss = hinge_loss_discriminator(real_output, fake_output)

                # Generator losses
                g_adv_loss = hinge_loss_generator(fake_output)
                g_l1_loss = self.l1_loss(fake_images, real_images)
                g_perc_loss = self.perceptual_loss(fake_images, real_images)
                g_loss = (g_adv_loss +
                         self.args.lambda_l1 * g_l1_loss +
                         self.args.lambda_perceptual * g_perc_loss)

                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                total_l1_loss += g_l1_loss.item()
                num_batches += 1

        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches

        self.generator.train()
        self.discriminator.train()

        return {
            'val_g_loss': avg_g_loss,
            'val_d_loss': avg_d_loss,
            'val_l1_loss': avg_l1_loss
        }

    def _save_samples(self, epoch: int):
        """Save sample images"""
        self.generator.eval()

        with torch.no_grad():
            # Get a batch from validation set
            edges, attributes, real_images = next(iter(self.val_loader))
            edges = edges[:8].to(self.device)  # Take first 8 samples
            attributes = attributes[:8].to(self.device)
            real_images = real_images[:8].to(self.device)

            # Generate fake images
            fake_images = self.generator(edges, attributes)

            # Denormalize for visualization
            edges_vis = (edges + 1) / 2
            real_vis = (real_images + 1) / 2
            fake_vis = (fake_images + 1) / 2

            # Create comparison grid
            comparison = torch.cat([edges_vis, real_vis, fake_vis], dim=0)  # [24, 3, H, W]

            # Save grid
            save_path = os.path.join(self.args.sample_dir, f'epoch_{epoch:03d}.png')
            vutils.save_image(
                comparison,
                save_path,
                nrow=8,  # 8 images per row
                normalize=False
            )

        self.generator.train()

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'g_scaler_state_dict': self.g_scaler.state_dict(),
            'd_scaler_state_dict': self.d_scaler.state_dict(),
            'args': self.args,
            'attr_dim': self.attr_dim,
            'dataset_meta': self.dataset_meta,
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)

        # Save epoch checkpoint
        if epoch % self.args.save_freq == 0:
            epoch_path = os.path.join(self.args.checkpoint_dir, f'epoch_{epoch:03d}.pth')
            torch.save(checkpoint, epoch_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

        if 'g_scheduler_state_dict' in checkpoint:
            self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        if 'd_scheduler_state_dict' in checkpoint:
            self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])

        # Restore AMP scalers
        if 'g_scaler_state_dict' in checkpoint:
            self.g_scaler.load_state_dict(checkpoint['g_scaler_state_dict'])
        if 'd_scaler_state_dict' in checkpoint:
            self.d_scaler.load_state_dict(checkpoint['d_scaler_state_dict'])

        # Restore dataset metadata
        if 'dataset_meta' in checkpoint:
            saved_meta = checkpoint['dataset_meta']
            print(f"  Restored {len(saved_meta.get('attribute_columns', []))} attribute columns")

        print(f"Resuming from epoch {self.current_epoch}")

    def train(self):
        """Main training loop"""
        print("Starting training...")
        start_time = time.time()

        for epoch in range(self.current_epoch, self.args.epochs):
            epoch_start_time = time.time()

            # Training
            train_stats = self._train_epoch(epoch)

            # Validation
            if epoch % self.args.val_freq == 0:
                val_stats = self._validate()
            else:
                val_stats = {}

            # Update learning rates
            self.g_scheduler.step()
            self.d_scheduler.step()

            # Save samples
            if epoch % self.args.sample_freq == 0:
                self._save_samples(epoch)

            # Save checkpoint
            is_best = False
            if val_stats and val_stats['val_g_loss'] < self.best_loss:
                self.best_loss = val_stats['val_g_loss']
                is_best = True

            self._save_checkpoint(epoch, is_best)

            # Logging
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch:03d}/{self.args.epochs:03d} ({epoch_time:.1f}s)")
            print(f"  G: {train_stats['g_loss']:.4f}, D: {train_stats['d_loss']:.4f}")
            if val_stats:
                print(f"  Val G: {val_stats['val_g_loss']:.4f}, Val D: {val_stats['val_d_loss']:.4f}")

            # Tensorboard logging
            for key, value in train_stats.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_stats.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)

            self.writer.add_scalar('lr/generator', self.g_optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('lr/discriminator', self.d_optimizer.param_groups[0]['lr'], epoch)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")
        self.writer.close()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0
        total_d_loss = 0
        total_g_adv_loss = 0
        total_g_l1_loss = 0
        total_g_perc_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:03d}')

        for edges, attributes, real_images in pbar:
            edges = edges.to(self.device)
            attributes = attributes.to(self.device)
            real_images = real_images.to(self.device)

            # Train Generator
            g_stats, fake_images = self._train_generator(edges, attributes, real_images)

            # Train Discriminator
            d_stats = self._train_discriminator(edges, attributes, real_images, fake_images)

            # Update statistics
            total_g_loss += g_stats['g_loss']
            total_d_loss += d_stats['d_loss']
            total_g_adv_loss += g_stats['g_adv_loss']
            total_g_l1_loss += g_stats['g_l1_loss']
            total_g_perc_loss += g_stats['g_perc_loss']
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'G': f"{g_stats['g_loss']:.3f}",
                'D': f"{d_stats['d_loss']:.3f}",
                'L1': f"{g_stats['g_l1_loss']:.3f}"
            })

            self.global_step += 1

        # Return average losses
        return {
            'g_loss': total_g_loss / num_batches,
            'd_loss': total_d_loss / num_batches,
            'g_adv_loss': total_g_adv_loss / num_batches,
            'g_l1_loss': total_g_l1_loss / num_batches,
            'g_perc_loss': total_g_perc_loss / num_batches
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Conditional GAN for Shoe Generation')

    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the Zappos dataset')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Size of input images')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--min_attr_freq', type=int, default=100,
                       help='Minimum frequency for attributes to be included')
    parser.add_argument('--attr_prefixes', type=str, default=None,
                       help='Comma-separated attribute prefixes to use (e.g., "Category,Material")')

    # Model arguments
    parser.add_argument('--ngf', type=int, default=64,
                       help='Number of generator filters')
    parser.add_argument('--ndf', type=int, default=64,
                       help='Number of discriminator filters')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--g_lr', type=float, default=0.0002,
                       help='Learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0002,
                       help='Learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Beta2 for Adam optimizer')

    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                       help='Weight for L1 loss')
    parser.add_argument('--lambda_perceptual', type=float, default=10.0,
                       help='Weight for perceptual loss')

    # Output directories
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--sample_dir', type=str, default='./samples',
                       help='Directory to save sample images')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save logs')

    # Miscellaneous
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Frequency of saving checkpoints')
    parser.add_argument('--sample_freq', type=int, default=5,
                       help='Frequency of saving sample images')
    parser.add_argument('--val_freq', type=int, default=1,
                       help='Frequency of validation')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()