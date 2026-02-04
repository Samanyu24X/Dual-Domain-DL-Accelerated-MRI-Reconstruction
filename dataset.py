"""
Dataset and DataLoader for MRI Image Reconstruction
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class MRIDataset(Dataset):
    """
    PyTorch Dataset for MRI reconstruction from k-space data

    Args:
        data_dir: Directory containing .h5 files
        transform: Optional transform to apply to the data
        use_mask: Whether to apply undersampling mask to create input (default: True)
    """

    def __init__(self, data_dir, transform=None, use_mask=True, crop_size=(640, 320)):
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob('*.h5')))
        self.transform = transform
        self.use_mask = use_mask
        self.crop_size = crop_size

        if len(self.files) == 0:
            raise ValueError(f"No .h5 files found in {data_dir}")

        print(f"Loaded {len(self.files)} files from {data_dir}")

        # Precompute slice indices for faster access
        self.slice_indices = []
        for file_idx, file_path in enumerate(self.files):
            with h5py.File(file_path, 'r') as f:
                # Get number of slices from kspace
                if 'kspace' in f:
                    num_slices = f['kspace'].shape[0]
                    for slice_idx in range(num_slices):
                        self.slice_indices.append((file_idx, slice_idx))

        print(f"Total slices: {len(self.slice_indices)}")

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_indices[idx]
        file_path = self.files[file_idx]

        with h5py.File(file_path, 'r') as f:
            # Load k-space data
            kspace = f['kspace'][slice_idx]  # Shape: (H, W), complex64

            # Load mask if available
            mask = f['mask'][()] if 'mask' in f else None

            # Target: Fully-sampled reconstruction (inverse FFT of full k-space)
            target = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
            target = np.abs(target)

            # Input: Undersampled reconstruction
            if self.use_mask and mask is not None:
                # Apply mask to k-space
                kspace_undersampled = kspace * mask
            else:
                # If no mask, use full k-space (for testing purposes)
                kspace_undersampled = kspace

            # Zero-filled reconstruction (inverse FFT of undersampled k-space)
            input_data = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace_undersampled)))
            input_data = np.abs(input_data)

        # Normalize
        input_data = self._normalize(input_data)
        target = self._normalize(target)

        # Center crop to fixed size
        input_data = self._center_crop(input_data, self.crop_size)
        target = self._center_crop(target, self.crop_size)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)  # Add channel dimension
        target_tensor = torch.from_numpy(target).float().unsqueeze(0)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor

    def _normalize(self, data):
        """Normalize data to [0, 1] range"""
        data = data.astype(np.float32)
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 0:
            data = (data - data_min) / (data_max - data_min)
        return data

    def _center_crop(self, data, crop_size):
        """Center crop data to specified size"""
        h, w = data.shape
        target_h, target_w = crop_size

        # If data is smaller than crop size, pad it
        if h < target_h or w < target_w:
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            data = np.pad(data, ((pad_h // 2, pad_h - pad_h // 2),
                                 (pad_w // 2, pad_w - pad_w // 2)), mode='constant')
            h, w = data.shape

        # Center crop
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return data[start_h:start_h + target_h, start_w:start_w + target_w]


def create_data_loaders(data_dir='data', batch_size=8, num_workers=4):
    """
    Create train, validation, and test data loaders

    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    data_path = Path(data_dir)

    # Create datasets
    train_dataset = MRIDataset(data_path / 'train')
    val_dataset = MRIDataset(data_path / 'val')
    test_dataset = MRIDataset(data_path / 'test')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing MRIDataset...")

    # First, you need to run split_dataset.py to create the data splits
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir='data',
            batch_size=4,
            num_workers=0  # Use 0 for testing
        )

        print("\nDataLoader Statistics:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Test one batch
        print("\nTesting one batch...")
        for inputs, targets in train_loader:
            print(f"  Input shape: {inputs.shape}")
            print(f"  Target shape: {targets.shape}")
            print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
            break

    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease run 'python split_dataset.py' first to create the data splits.")
