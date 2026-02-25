"""
Colab/Kaggle Training Wrapper for Pix2Pix GAN
=============================================

Usage in Google Colab or Kaggle:

1. Upload this file + train.py + models.py + dataset.py + requirements.txt
2. Upload your dataset zip (ut-zap50k-images-square.zip)
3. Run this script

This wrapper:
- Mounts Google Drive for persistent checkpoint storage (Colab)
- Auto-resumes from the latest checkpoint if disconnected
- Saves checkpoints every 5 epochs to Drive
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


def install_deps():
    """Install required packages"""
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q',
        'torch', 'torchvision', 'numpy', 'opencv-python',
        'scipy', 'pandas', 'matplotlib', 'tqdm', 'Pillow', 'tensorboard'
    ], check=True)


def clone_repo(work_dir):
    """Clone the repo if not already present"""
    repo_dir = os.path.join(work_dir, 'jisha-pix2pix-gan')
    if not os.path.exists(repo_dir):
        subprocess.run([
            'git', 'clone',
            'https://github.com/jrajala6/jisha-pix2pix-gan.git',
            repo_dir
        ], check=True)
    return repo_dir


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint to resume from"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None

    # Extract epoch numbers and find the latest
    def get_epoch(path):
        basename = os.path.basename(path)
        try:
            return int(basename.split('_')[-1].replace('.pth', ''))
        except ValueError:
            return -1

    checkpoints.sort(key=get_epoch)
    latest = checkpoints[-1]
    print(f"Found checkpoint to resume from: {latest}")
    return latest


def sync_checkpoints_to_drive(local_dir, drive_dir):
    """Copy checkpoints to Google Drive for persistence"""
    if drive_dir and os.path.exists(drive_dir):
        for f in glob.glob(os.path.join(local_dir, '*.pth')):
            dst = os.path.join(drive_dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                print(f"Synced {os.path.basename(f)} to Drive")


def main():
    platform = detect_platform()
    print(f"Platform detected: {platform}")
    print("=" * 50)

    # Platform-specific setup
    if platform == 'colab':
        print("Setting up Google Colab...")
        drive_dir = setup_colab()
        work_dir = '/content'
    elif platform == 'kaggle':
        print("Setting up Kaggle...")
        drive_dir = None
        work_dir = '/kaggle/working'
        setup_kaggle()
    else:
        print("Running locally...")
        drive_dir = None
        work_dir = '.'

    # Install dependencies
    print("\nInstalling dependencies...")
    install_deps()

    # Clone repo
    print("\nCloning repository...")
    repo_dir = clone_repo(work_dir)
    os.chdir(repo_dir)
    print(f"Working directory: {os.getcwd()}")

    # Check for dataset
    dataset_dir = 'ut-zap50k-images-square'
    if not os.path.exists(dataset_dir):
        # Check for zip file in common locations
        zip_locations = [
            'ut-zap50k-images-square.zip',
            '../ut-zap50k-images-square.zip',
            '/content/ut-zap50k-images-square.zip',
            '/kaggle/input/ut-zap50k-images-square/ut-zap50k-images-square.zip',
        ]
        if drive_dir:
            zip_locations.append(f'{drive_dir}/ut-zap50k-images-square.zip')

        found = False
        for zip_path in zip_locations:
            if os.path.exists(zip_path):
                print(f"\nExtracting dataset from {zip_path}...")
                subprocess.run(['unzip', '-q', zip_path, '-d', '.'], check=True)
                found = True
                break

        # Also check if dataset is a Kaggle dataset
        kaggle_input = '/kaggle/input'
        if not found and os.path.exists(kaggle_input):
            for d in os.listdir(kaggle_input):
                potential = os.path.join(kaggle_input, d, 'ut-zap50k-images-square')
                if os.path.exists(potential):
                    os.symlink(potential, dataset_dir)
                    found = True
                    print(f"Linked Kaggle dataset from {potential}")
                    break

        if not found:
            print("\n⚠️  Dataset not found! Please upload ut-zap50k-images-square.zip")
            print("  For Colab: Upload to /content/ or Google Drive")
            print("  For Kaggle: Add as a dataset input")
            return

    print(f"\n✅ Dataset found: {dataset_dir}")

    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"✅ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("⚠️  No GPU detected! Training will be very slow.")

    # Check for existing checkpoints to resume
    checkpoint_dir = './checkpoints'
    resume_from = None

    # First check drive for checkpoints (Colab)
    if drive_dir:
        drive_ckpt_dir = f'{drive_dir}/checkpoints'
        resume_from = find_latest_checkpoint(drive_ckpt_dir)
        if resume_from:
            # Copy checkpoint locally
            local_copy = os.path.join(checkpoint_dir, os.path.basename(resume_from))
            os.makedirs(checkpoint_dir, exist_ok=True)
            shutil.copy2(resume_from, local_copy)
            resume_from = local_copy

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
        print(f"\n🔄 Resuming from: {resume_from}")

    print(f"\n{'='*50}")
    print("Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")

    # Run training
    process = subprocess.Popen(cmd)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nTraining interrupted! Checkpoints are saved.")
        process.terminate()
    finally:
        # Sync checkpoints to Drive (Colab)
        if drive_dir:
            print("\nSyncing checkpoints to Google Drive...")
            sync_checkpoints_to_drive(checkpoint_dir, f'{drive_dir}/checkpoints')
            sync_checkpoints_to_drive('./samples', f'{drive_dir}/samples')
            print("✅ Checkpoints synced to Drive!")

    print("\n✅ Training complete!")
    if drive_dir:
        print(f"Checkpoints saved to: {drive_dir}/checkpoints/")
        print(f"Samples saved to: {drive_dir}/samples/")


if __name__ == '__main__':
    main()
