import os
import pandas as pd
import numpy as np
import cv2
from scipy.io import loadmat
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ZapposDataset(Dataset):
    """
    Dataset loader for UT Zappos50K dataset with edge map generation and attribute conditioning.

    Returns:
        edges: Edge map tensor [C, H, W]
        attributes: Multi-hot attribute vector [attr_dim]
        image: RGB image tensor [3, H, W]
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        split: str = 'train',
        train_ratio: float = 0.8,
        min_attribute_freq: int = 100,
        attribute_prefixes: Optional[List[str]] = None,
        edge_threshold1: int = 50,
        edge_threshold2: int = 150,
        attribute_columns: Optional[List[str]] = None
    ):
        """
        Args:
            data_root: Path to dataset root containing 'image-path.mat', 'meta-data-bin.csv', 'images/'
            image_size: Target image size for resizing
            split: 'train' or 'val'
            train_ratio: Ratio of data to use for training
            min_attribute_freq: Minimum frequency to keep an attribute
            attribute_prefixes: List of attribute prefixes to use (e.g., ['Category', 'Material'])
            edge_threshold1: Lower threshold for Canny edge detection
            edge_threshold2: Upper threshold for Canny edge detection
        """
        self.data_root = data_root
        self.image_size = image_size
        self.split = split
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2

        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])

        self.edge_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Load and process data
        self._load_image_paths()
        self._filter_missing_images()
        self._load_attributes(min_attribute_freq, attribute_prefixes, attribute_columns)
        self._create_split(train_ratio)

        print(f"Loaded {len(self.indices)} samples for {split} split")
        print(f"Attribute vector dimension: {self.attr_dim}")
        print(f"Using attribute columns: {self.attribute_columns[:10]}...")  # Show first 10

    def _load_image_paths(self):
        """Load image paths from .mat file"""
        # Look for image-path.mat in ut-zap50k-data subdirectory
        mat_path = os.path.join(self.data_root, 'ut-zap50k-data', 'image-path.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"image-path.mat not found at {mat_path}")

        try:
            mat_data = loadmat(mat_path)
            # Handle different possible structures in .mat file
            if 'imagepath' in mat_data:
                paths = mat_data['imagepath']
            elif 'imagePath' in mat_data:
                paths = mat_data['imagePath']
            elif 'image_path' in mat_data:
                paths = mat_data['image_path']
            else:
                # Try to find the first non-metadata key
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if keys:
                    paths = mat_data[keys[0]]
                else:
                    raise KeyError("Could not find image paths in .mat file")

            # Convert to list of strings
            if paths.dtype == 'object':
                self.image_paths = []
                for path in paths.flatten():
                    if isinstance(path, np.ndarray) and len(path) > 0:
                        path_str = str(path[0])
                    else:
                        path_str = str(path)
                    self.image_paths.append(path_str)
            else:
                self.image_paths = [str(path) for path in paths.flatten()]

            print(f"Loaded {len(self.image_paths)} image paths")
            print(f"Sample paths: {self.image_paths[:3]}")

        except Exception as e:
            raise Exception(f"Error loading image paths: {e}")

    def _filter_missing_images(self):
        """Remove entries where the image file doesn't exist on disk"""
        valid_indices = []
        missing = 0
        for i, rel_path in enumerate(self.image_paths):
            full_path = os.path.join(self.data_root, rel_path)
            if os.path.exists(full_path):
                valid_indices.append(i)
            else:
                missing += 1

        if missing > 0:
            self.image_paths = [self.image_paths[i] for i in valid_indices]
            self.valid_indices = valid_indices  # Track for attribute alignment
            print(f"Filtered out {missing} missing images ({100*missing/(missing+len(valid_indices)):.1f}%), {len(self.image_paths)} remaining")
        else:
            self.valid_indices = None
            print("All image paths verified — no missing files")

    def _load_attributes(self, min_freq: int, prefixes: Optional[List[str]], fixed_columns: Optional[List[str]] = None):
        """Load and filter attributes from CSV file"""
        # Look for meta-data-bin.csv in ut-zap50k-data subdirectory
        csv_path = os.path.join(self.data_root, 'ut-zap50k-data', 'meta-data-bin.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"meta-data-bin.csv not found at {csv_path}")

        try:
            self.attributes_df = pd.read_csv(csv_path)
            print(f"Loaded attributes CSV with {len(self.attributes_df)} rows and {len(self.attributes_df.columns)} columns")
        except Exception as e:
            raise Exception(f"Error loading attributes CSV: {e}")

        # Align attributes with filtered image paths
        if self.valid_indices is not None:
            self.attributes_df = self.attributes_df.iloc[self.valid_indices].reset_index(drop=True)

        # Ensure we have the same number of samples
        n_images = len(self.image_paths)
        n_attributes = len(self.attributes_df)

        if n_images != n_attributes:
            print(f"Warning: Mismatch between images ({n_images}) and attributes ({n_attributes})")
            min_len = min(n_images, n_attributes)
            self.image_paths = self.image_paths[:min_len]
            self.attributes_df = self.attributes_df.iloc[:min_len]

        # Use fixed columns if provided (e.g. val reusing train's columns)
        if fixed_columns is not None:
            self.attribute_columns = fixed_columns
            self.attr_dim = len(self.attribute_columns)
            print(f"Using {self.attr_dim} pre-set attribute columns")
            return

        # Get all columns except CID (first column)
        all_columns = list(self.attributes_df.columns)[1:]  # Skip first column (CID)

        # Filter columns by prefixes if specified
        if prefixes:
            selected_cols = []
            for prefix in prefixes:
                cols = [col for col in all_columns if col.startswith(prefix)]
                selected_cols.extend(cols)

            if not selected_cols:
                print(f"Warning: No columns found for prefixes {prefixes}, using all attribute columns")
                selected_cols = all_columns
        else:
            # Use all attribute columns (excluding CID)
            selected_cols = all_columns

        # Filter by frequency if min_freq > 0
        if min_freq > 0:
            frequent_cols = []
            for col in selected_cols:
                if col in self.attributes_df.columns:
                    freq = self.attributes_df[col].sum()
                    if freq >= min_freq:
                        frequent_cols.append(col)

            if frequent_cols:
                selected_cols = frequent_cols
                print(f"Filtered to {len(frequent_cols)} attributes with frequency >= {min_freq}")
            else:
                print(f"Warning: No attributes meet frequency threshold {min_freq}, using all")

        self.attribute_columns = selected_cols
        self.attr_dim = len(self.attribute_columns)

        if self.attr_dim == 0:
            raise ValueError("No valid attribute columns found")

        print(f"Selected {self.attr_dim} attribute columns")
        print(f"Sample attributes: {self.attribute_columns[:5]}")

    def _create_split(self, train_ratio: float):
        """Create train/val split"""
        total_samples = len(self.image_paths)
        indices = np.arange(total_samples)
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(indices)

        split_idx = int(train_ratio * total_samples)

        if self.split == 'train':
            self.indices = indices[:split_idx]
        else:  # val
            self.indices = indices[split_idx:]

    def _generate_edge_map(self, image: np.ndarray) -> np.ndarray:
        """Generate edge map using Canny edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.edge_threshold1, self.edge_threshold2)

        # Convert to 3-channel for consistency
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return edges_3ch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            edges: Edge map tensor [3, H, W]
            attributes: Attribute vector [attr_dim]
            image: RGB image tensor [3, H, W]
        """
        real_idx = self.indices[idx]

        # Load image - combine data_root with relative path from .mat file
        relative_path = self.image_paths[real_idx]
        img_path = os.path.join(self.data_root, relative_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            # Shouldn't happen after pre-filtering, but fallback to a random valid sample
            return self.__getitem__((idx + 1) % len(self))

        # Convert to numpy for edge detection
        image_np = np.array(image)

        # Generate edge map
        edges_np = self._generate_edge_map(image_np)

        # Apply transforms
        image_tensor = self.image_transform(image)
        edges_tensor = self.edge_transform(edges_np)

        # Normalize edge map to [-1, 1] to match image normalization
        edges_tensor = edges_tensor * 2.0 - 1.0

        # Load attributes
        try:
            attr_row = self.attributes_df.iloc[real_idx]
            attr_vector = attr_row[self.attribute_columns].values.astype(np.float32)
        except (IndexError, KeyError):
            print(f"Warning: Could not load attributes for index {real_idx}, using zeros")
            attr_vector = np.zeros(self.attr_dim, dtype=np.float32)

        attr_tensor = torch.from_numpy(attr_vector)

        return edges_tensor, attr_tensor, image_tensor


def create_dataloader(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 256,
    train_ratio: float = 0.8,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Returns:
        train_loader, val_loader
    """

    # Create datasets (val reuses train's attribute columns for consistency)
    train_dataset = ZapposDataset(
        data_root=data_root,
        image_size=image_size,
        split='train',
        train_ratio=train_ratio,
        **kwargs
    )

    val_dataset = ZapposDataset(
        data_root=data_root,
        image_size=image_size,
        split='val',
        train_ratio=train_ratio,
        attribute_columns=train_dataset.attribute_columns,
        **kwargs
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    import matplotlib.pyplot as plt

    # Use the actual data path
    data_root = "./ut-zap50k-images-square"

    try:
        dataset = ZapposDataset(data_root, image_size=256)
        print(f"Dataset loaded successfully with {len(dataset)} samples")

        # Test loading a sample
        edges, attrs, image = dataset[0]
        print(f"Edges shape: {edges.shape}")
        print(f"Attributes shape: {attrs.shape}")
        print(f"Image shape: {image.shape}")
        print(f"Attributes sum: {attrs.sum()}")

        # Test multiple samples to verify everything works
        print("\nTesting multiple samples:")
        for i in range(min(3, len(dataset))):
            edges, attrs, image = dataset[i]
            print(f"Sample {i}: edges {edges.shape}, attrs sum: {attrs.sum():.1f}")

    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure to set the correct data_root path")