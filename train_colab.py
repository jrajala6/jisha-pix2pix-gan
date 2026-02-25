"""
Colab/Kaggle Training Wrapper for Pix2Pix GAN
=============================================

Usage in Google Colab:

Cell 1:
    from google.colab import drive
    drive.mount('/content/drive')

Cell 2:
    !git clone https://github.com/jrajala6/jisha-pix2pix-gan.git
    %cd jisha-pix2pix-gan
    !python train_colab.py

This wrapper:
- Uses Google Drive for persistent checkpoint storage (Colab)
- Auto-resumes from latest.pth if session disconnects
- Syncs checkpoints + samples to Drive after training
"""

import os
import sys
import subprocess
import shutil
import glob


def detect_platform():
    """Detect if running on Colab, Kaggle, or local"""
    if 'COLAB_GPU' in os.environ or os.path.exists('/content'):
        return 'colab'
    elif os.path.exists('/kaggle'):
        return 'kaggle'
    else:
        return 'local'


def setup_colab():
    """Set up Google Drive workspace (Drive must be mounted from notebook first)"""
    drive_path = '/content/drive/MyDrive'

    if os.path.exists(drive_path):
        print("  ✅ Google Drive is mounted!")
        drive_dir = f'{drive_path}/pix2pix-gan'
        os.makedirs(drive_dir, exist_ok=True)
        os.makedirs(f'{drive_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{drive_dir}/samples', exist_ok=True)
        os.makedirs(f'{drive_dir}/logs', exist_ok=True)
        return drive_dir
    else:
        print("  ⚠️  Google Drive not mounted. Checkpoints will be saved locally only.")
        print("  To save to Drive, run this in a notebook cell BEFORE this script:")
        print("    from google.colab import drive")
        print("    drive.mount('/content/drive')")
        return None


def setup_kaggle():
    """Set up Kaggle workspace"""
    work_dir = '/kaggle/working'
    os.makedirs(f'{work_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{work_dir}/samples', exist_ok=True)
    os.makedirs(f'{work_dir}/logs', exist_ok=True)
    return work_dir


def install_deps(platform):
    """Install required packages (skip torch on Colab — already CUDA-matched)"""
    if platform == 'colab':
        # Colab already has CUDA-matched PyTorch; only install extras
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q',
            'opencv-python', 'scipy', 'pandas', 'matplotlib', 'tqdm', 'Pillow', 'tensorboard'
        ], check=True)
    else:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q',
            'torch', 'torchvision', 'numpy', 'opencv-python',
            'scipy', 'pandas', 'matplotlib', 'tqdm', 'Pillow', 'tensorboard'
        ], check=True)


def clone_repo(work_dir):
    """Clone the repo, or pull latest if already cloned"""
    repo_dir = os.path.join(work_dir, 'jisha-pix2pix-gan')
    if os.path.exists(repo_dir):
        print("  Repo exists, pulling latest changes...")
        subprocess.run(['git', 'pull'], cwd=repo_dir, check=True)
    else:
        subprocess.run([
            'git', 'clone',
            'https://github.com/jrajala6/jisha-pix2pix-gan.git',
            repo_dir
        ], check=True)
    return repo_dir


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint to resume from.

    Priority: latest.pth > highest epoch_XXX.pth > best.pth
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    # First try latest.pth (always the most recent)
    latest = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(latest):
        print(f"  Found latest.pth to resume from")
        return latest

    # Then try epoch files (epoch_000.pth, epoch_005.pth, ...)
    epoch_files = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
    if epoch_files:
        def get_epoch_num(path):
            basename = os.path.basename(path)
            try:
                return int(basename.replace('epoch_', '').replace('.pth', ''))
            except ValueError:
                return -1

        epoch_files.sort(key=get_epoch_num)
        best_epoch = epoch_files[-1]
        print(f"  Found {os.path.basename(best_epoch)} to resume from")
        return best_epoch

    # Last resort: best.pth
    best = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best):
        print(f"  Found best.pth to resume from")
        return best

    return None


def sync_files(src_dir, dst_dir, patterns=None):
    """Copy files matching patterns from src to dst for persistence.

    Args:
        src_dir: Local directory to copy from
        dst_dir: Drive directory to copy to
        patterns: List of glob patterns (default: common training outputs)
    """
    if not dst_dir or not os.path.exists(src_dir):
        return

    if patterns is None:
        patterns = ['*.pth', '*.png', '*.jpg', 'events*']

    os.makedirs(dst_dir, exist_ok=True)
    synced = 0
    for pattern in patterns:
        for f in glob.glob(os.path.join(src_dir, pattern)):
            dst = os.path.join(dst_dir, os.path.basename(f))
            # Always overwrite latest.pth and best.pth, skip others if exist
            basename = os.path.basename(f)
            if basename in ('latest.pth', 'best.pth') or not os.path.exists(dst):
                shutil.copy2(f, dst)
                synced += 1

    if synced > 0:
        print(f"  Synced {synced} files to {dst_dir}")


def check_dataset(dataset_dir, drive_dir=None):
    """Check that both images and metadata exist. Returns True if ready."""
    images_ok = any(
        os.path.isdir(os.path.join(dataset_dir, d))
        for d in ['Boots', 'Sandals', 'Shoes', 'Slippers']
    )
    metadata_dir = os.path.join(dataset_dir, 'ut-zap50k-data')
    metadata_ok = (
        os.path.exists(os.path.join(metadata_dir, 'image-path.mat')) and
        os.path.exists(os.path.join(metadata_dir, 'meta-data-bin.csv'))
    )

    if images_ok and metadata_ok:
        return True

    # Try to find and extract zip
    zip_locations = [
        'ut-zap50k-images-square.zip',
        '../ut-zap50k-images-square.zip',
        '/content/ut-zap50k-images-square.zip',
        '/kaggle/input/ut-zap50k-images-square/ut-zap50k-images-square.zip',
    ]
    if drive_dir:
        zip_locations.append(f'{drive_dir}/ut-zap50k-images-square.zip')

    for zip_path in zip_locations:
        if os.path.exists(zip_path):
            print(f"  Extracting dataset from {zip_path}...")
            subprocess.run(['unzip', '-q', '-o', zip_path, '-d', '.'], check=True)
            # Re-check after extraction
            return check_dataset(dataset_dir, drive_dir=None)

    # Check Kaggle inputs
    kaggle_input = '/kaggle/input'
    if os.path.exists(kaggle_input):
        for d in os.listdir(kaggle_input):
            potential = os.path.join(kaggle_input, d, 'ut-zap50k-images-square')
            if os.path.exists(potential):
                if not os.path.exists(dataset_dir):
                    os.symlink(potential, dataset_dir)
                print(f"  Linked Kaggle dataset from {potential}")
                return check_dataset(dataset_dir, drive_dir=None)

    # Print what's missing
    if not images_ok:
        print("  ❌ Image folders (Boots/Sandals/Shoes/Slippers) not found")
    if not metadata_ok:
        print("  ❌ Metadata (ut-zap50k-data/image-path.mat, meta-data-bin.csv) not found")
    print("\n  Please upload ut-zap50k-images-square.zip containing both images and ut-zap50k-data/")

    return False


def main():
    platform = detect_platform()
    print(f"Platform: {platform}")
    print("=" * 50)

    # Platform-specific setup
    drive_dir = None
    if platform == 'colab':
        print("Setting up Google Colab...")
        drive_dir = setup_colab()
        work_dir = '/content'
    elif platform == 'kaggle':
        print("Setting up Kaggle...")
        work_dir = '/kaggle/working'
        setup_kaggle()
    else:
        print("Running locally...")
        work_dir = '.'

    # Install dependencies (skip torch on Colab)
    print("\nInstalling dependencies...")
    install_deps(platform)

    # Clone repo
    print("\nCloning repository...")
    repo_dir = clone_repo(work_dir)
    os.chdir(repo_dir)
    print(f"Working directory: {os.getcwd()}")

    # Check dataset (images + metadata)
    dataset_dir = 'ut-zap50k-images-square'
    print(f"\nChecking dataset...")
    if not check_dataset(dataset_dir, drive_dir):
        return

    print(f"✅ Dataset ready: {dataset_dir}")

    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("⚠️  No GPU detected! Training will be very slow.")

    # Check for existing checkpoints to resume
    checkpoint_dir = './checkpoints'
    resume_from = None

    # First check Drive for checkpoints (Colab)
    if drive_dir:
        drive_ckpt_dir = f'{drive_dir}/checkpoints'
        drive_ckpt = find_latest_checkpoint(drive_ckpt_dir)
        if drive_ckpt:
            # Copy checkpoint locally for resume
            os.makedirs(checkpoint_dir, exist_ok=True)
            local_copy = os.path.join(checkpoint_dir, os.path.basename(drive_ckpt))
            shutil.copy2(drive_ckpt, local_copy)
            resume_from = local_copy
            print(f"  Copied from Drive: {os.path.basename(drive_ckpt)}")

    # Also check local checkpoints
    if not resume_from:
        resume_from = find_latest_checkpoint(checkpoint_dir)

    # Build training command
    cmd = [
        sys.executable, 'train.py',
        '--data_root', f'./{dataset_dir}',
        '--epochs', '30',
        '--batch_size', '32',
        '--num_workers', '4',
        '--image_size', '256',
        '--save_freq', '5',
        '--sample_freq', '5',
        '--checkpoint_dir', checkpoint_dir,
    ]

    if resume_from:
        cmd.extend(['--resume_from', resume_from])
        print(f"\n🔄 Resuming from: {os.path.basename(resume_from)}")

    print(f"\n{'='*50}")
    print("Starting training...")
    print(f"{'='*50}\n")

    # Run training
    process = subprocess.Popen(cmd)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nTraining interrupted! Checkpoints are saved.")
        process.terminate()
    finally:
        # Sync to Drive (Colab)
        if drive_dir:
            print("\nSyncing to Google Drive...")
            sync_files(checkpoint_dir, f'{drive_dir}/checkpoints', ['*.pth'])
            sync_files('./samples', f'{drive_dir}/samples', ['*.png', '*.jpg'])
            sync_files('./logs', f'{drive_dir}/logs', ['events*'])
            print("✅ All outputs synced to Drive!")

    print("\n✅ Training complete!")
    if drive_dir:
        print(f"  Checkpoints: {drive_dir}/checkpoints/")
        print(f"  Samples: {drive_dir}/samples/")
        print(f"  Logs: {drive_dir}/logs/")


if __name__ == '__main__':
    main()
