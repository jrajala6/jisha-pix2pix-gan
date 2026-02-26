"""
Colab/Kaggle Training Wrapper for Pix2Pix GAN (FiLM variant)
=============================================================

Colab usage:

Cell 1:
    from google.colab import drive
    drive.mount('/content/drive')

Cell 2:
    !git clone https://github.com/jrajala6/jisha-pix2pix-gan.git
    %cd jisha-pix2pix-gan
    !python train_colab_film.py

This wrapper:
- Uses Google Drive for persistent checkpoint storage (Colab)
- Auto-resumes from latest.pth if session disconnects
- Handles nested dataset extraction folders
- Saves to pix2pix-gan-film/ (separate from baseline)
"""

import os
import sys
import subprocess
import shutil
import glob


def detect_platform():
    if 'COLAB_GPU' in os.environ or os.path.exists('/content'):
        return 'colab'
    elif os.path.exists('/kaggle'):
        return 'kaggle'
    else:
        return 'local'


def setup_colab():
    drive_path = '/content/drive/MyDrive'
    if os.path.exists(drive_path):
        print("  ✅ Google Drive is mounted!")
        drive_dir = f'{drive_path}/pix2pix-gan-film'
        os.makedirs(drive_dir, exist_ok=True)
        os.makedirs(f'{drive_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{drive_dir}/samples', exist_ok=True)
        os.makedirs(f'{drive_dir}/logs', exist_ok=True)
        return drive_dir
    else:
        print("  ⚠️  Google Drive not mounted. Checkpoints will be saved locally only.")
        print("  Run this first in a notebook cell:")
        print("    from google.colab import drive")
        print("    drive.mount('/content/drive')")
        return None


def setup_kaggle():
    work_dir = '/kaggle/working'
    os.makedirs(f'{work_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{work_dir}/samples', exist_ok=True)
    os.makedirs(f'{work_dir}/logs', exist_ok=True)
    return work_dir


def install_deps(platform):
    if platform == 'colab':
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
    repo_dir = os.path.join(work_dir, 'jisha-pix2pix-gan')
    if os.path.exists(repo_dir):
        print("  Repo exists, pulling latest changes...")
        subprocess.run(['git', 'pull'], cwd=repo_dir, check=True)
    else:
        subprocess.run(['git', 'clone', 'https://github.com/jrajala6/jisha-pix2pix-gan.git', repo_dir], check=True)
    return repo_dir


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None

    latest = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(latest):
        print("  Found latest.pth to resume from")
        return latest

    epoch_files = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
    if epoch_files:
        def get_epoch_num(path):
            base = os.path.basename(path)
            try:
                return int(base.replace('epoch_', '').replace('.pth', ''))
            except ValueError:
                return -1
        epoch_files.sort(key=get_epoch_num)
        chosen = epoch_files[-1]
        print(f"  Found {os.path.basename(chosen)} to resume from")
        return chosen

    best = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best):
        print("  Found best.pth to resume from")
        return best

    return None


def dataset_ready(root):
    """Return True if root contains both image folders and ut-zap50k-data metadata."""
    images_ok = any(os.path.isdir(os.path.join(root, d)) for d in ['Boots', 'Sandals', 'Shoes', 'Slippers'])
    meta_dir = os.path.join(root, 'ut-zap50k-data')
    metadata_ok = (
        os.path.exists(os.path.join(meta_dir, 'image-path.mat')) and
        os.path.exists(os.path.join(meta_dir, 'meta-data-bin.csv'))
    )
    return images_ok and metadata_ok


def resolve_dataset_root(dataset_dir):
    """
    Handles cases like:
      ut-zap50k-images-square/ut-zap50k-images-square/ut-zap50k-data/...
    """
    if dataset_ready(dataset_dir):
        return dataset_dir

    nested = os.path.join(dataset_dir, 'ut-zap50k-images-square')
    if dataset_ready(nested):
        return nested

    if os.path.isdir(dataset_dir):
        for name in os.listdir(dataset_dir):
            cand = os.path.join(dataset_dir, name)
            if os.path.isdir(cand) and dataset_ready(cand):
                return cand

    return None


def check_and_extract_dataset(dataset_dir, drive_dir=None):
    """
    Ensures dataset exists; tries to unzip ut-zap50k-images-square.zip if missing.
    Returns resolved dataset root path or None.
    """
    resolved = resolve_dataset_root(dataset_dir)
    if resolved:
        return resolved

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
            resolved = resolve_dataset_root(dataset_dir)
            if resolved:
                return resolved

    print("  ❌ Dataset not found or incomplete.")
    print("  Expected image folders (Boots/Sandals/Shoes/Slippers) and ut-zap50k-data/{image-path.mat, meta-data-bin.csv}")
    print("  Upload ut-zap50k-images-square.zip (must include ut-zap50k-data/) to /content or Drive.")
    return None


def main():
    platform = detect_platform()
    print(f"Platform: {platform}")
    print("=" * 50)

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

    print("\nInstalling dependencies...")
    install_deps(platform)

    print("\nCloning repository...")
    repo_dir = clone_repo(work_dir)
    os.chdir(repo_dir)
    print(f"Working directory: {os.getcwd()}")

    dataset_dir = 'ut-zap50k-images-square'
    print("\nChecking dataset...")
    dataset_root = check_and_extract_dataset(dataset_dir, drive_dir)
    if not dataset_root:
        return
    print(f"✅ Dataset ready. Using dataset root: {dataset_root}")

    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("⚠️  No GPU detected! Training will be very slow.")

    if drive_dir:
        checkpoint_dir = f'{drive_dir}/checkpoints'
        sample_dir = f'{drive_dir}/samples'
        log_dir = f'{drive_dir}/logs'
        print("✅ Saving directly to Google Drive (disconnect-safe)")
    else:
        checkpoint_dir = './checkpoints'
        sample_dir = './samples'
        log_dir = './logs'
        print("⚠️  Saving locally only (session disconnect will lose data)")

    resume_from = find_latest_checkpoint(checkpoint_dir)

    # Use train_film.py (FiLM model with lambda_l1=50)
    train_script = 'train_film.py'
    print(f"Training script: {train_script}")

    cmd = [
        sys.executable, train_script,
        '--data_root', f'./{dataset_root}',
        '--epochs', '30',
        '--batch_size', '16',
        '--num_workers', '2',
        '--image_size', '256',
        '--lambda_l1', '50',
        '--lambda_perceptual', '10',
        '--save_freq', '5',
        '--sample_freq', '5',
        '--checkpoint_dir', checkpoint_dir,
        '--sample_dir', sample_dir,
        '--log_dir', log_dir,
    ]

    if resume_from:
        cmd.extend(['--resume_from', resume_from])
        print(f"\n🔄 Resuming from: {os.path.basename(resume_from)}")

    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    process = subprocess.Popen(cmd)
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nTraining interrupted! Checkpoints are saved.")
        process.terminate()

    print("\n✅ Training complete!")
    print(f"  Checkpoints: {checkpoint_dir}/")
    print(f"  Samples: {sample_dir}/")
    print(f"  Logs: {log_dir}/")


if __name__ == '__main__':
    main()
