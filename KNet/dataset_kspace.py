import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class MRIKspaceDataset(Dataset):
    """
    PyTorch Dataset for K-space domain MRI reconstruction
    
    Returns k-space data as 2-channel (real, imaginary) tensors
    """

    def __init__(self, data_dir, transform=None, use_mask=True, crop_size=(640, 320)):  # Same as I-Net
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob('*.h5')))
        self.transform = transform
        self.use_mask = use_mask
        self.crop_size = crop_size

        if len(self.files) == 0:
            raise ValueError(f"No .h5 files found in {data_dir}")

        print(f"Loaded {len(self.files)} files from {data_dir}")

        # Precompute slice indices
        self.slice_indices = []
        for file_idx, file_path in enumerate(self.files):
            with h5py.File(file_path, 'r') as f:
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
            # Load k-space data (complex valued)
            kspace = f['kspace'][slice_idx]  # Shape: (H, W), complex64
            mask = f['mask'][()] if 'mask' in f else None

            # Target: Full k-space
            target_kspace = kspace.copy()

        # Input: Undersampled k-space
        if self.use_mask and mask is not None:
            # Mask is 1D, need to broadcast it
            if len(mask.shape) == 1:
                # For vertical lines: reshape to (1, W) to broadcast across height
                mask = mask.reshape(1, -1)
            input_kspace = kspace * mask
        else:
            input_kspace = kspace.copy()

        # Center crop k-space FIRST
        input_kspace = self._center_crop_complex(input_kspace, self.crop_size)
        target_kspace = self._center_crop_complex(target_kspace, self.crop_size)

        # NEW: Better normalization - use percentile instead of max
        # This handles the huge dynamic range better
        scale = np.percentile(np.abs(target_kspace), 99)
        if scale > 0:
            input_kspace = input_kspace / scale
            target_kspace = target_kspace / scale

        # Convert complex to 2-channel (real, imaginary)
        input_tensor = self._complex_to_tensor(input_kspace)   # [2, H, W]
        target_tensor = self._complex_to_tensor(target_kspace) # [2, H, W]

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor

    def _complex_to_tensor(self, complex_array):
        """Convert complex array to 2-channel tensor [real, imag]"""
        real = np.real(complex_array).astype(np.float32)
        imag = np.imag(complex_array).astype(np.float32)
        return torch.stack([torch.from_numpy(real), torch.from_numpy(imag)], dim=0)

    def _center_crop_complex(self, data, crop_size):
        """Center crop complex-valued data"""
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


def create_kspace_data_loaders(data_dir='data', batch_size=8, num_workers=4):
    """Create train, validation, and test data loaders for k-space"""
    data_path = Path(data_dir)

    train_dataset = MRIKspaceDataset(data_path / 'train')
    val_dataset = MRIKspaceDataset(data_path / 'val')
    test_dataset = MRIKspaceDataset(data_path / 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing MRIKspaceDataset...")

    try:
        train_loader, val_loader, test_loader = create_kspace_data_loaders(
            data_dir='data',
            batch_size=4,
            num_workers=0
        )

        print("\nDataLoader Statistics:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        print("\nTesting one batch...")
        for inputs, targets in train_loader:
            print(f"  Input shape: {inputs.shape}")
            print(f"  Target shape: {targets.shape}")
            print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
            break

    except Exception as e:
        print(f"Error: {e}")