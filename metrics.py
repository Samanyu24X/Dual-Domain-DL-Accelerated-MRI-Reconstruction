"""
Evaluation metrics for MRI reconstruction
"""
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_nmse(pred, target):
    """
    Compute Normalized Mean Squared Error (NMSE)

    Args:
        pred: Predicted image (torch.Tensor)
        target: Ground truth image (torch.Tensor)

    Returns:
        NMSE value
    """
    mse = torch.mean((pred - target) ** 2)
    target_norm = torch.mean(target ** 2)
    nmse = mse / (target_norm + 1e-10)
    return nmse


def compute_psnr(pred, target, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)

    Args:
        pred: Predicted image (torch.Tensor)
        target: Ground truth image (torch.Tensor)
        max_val: Maximum possible pixel value (default: 1.0)

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-10))
    return psnr


def compute_ssim(pred, target, data_range=1.0):
    """
    Compute Structural Similarity Index (SSIM)

    Args:
        pred: Predicted image (torch.Tensor or numpy.ndarray)
        target: Ground truth image (torch.Tensor or numpy.ndarray)
        data_range: Range of the data (default: 1.0)

    Returns:
        SSIM value
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # If batched, compute SSIM for each image and average
    if pred.ndim == 4:  # (B, C, H, W)
        ssim_vals = []
        for i in range(pred.shape[0]):
            # Remove channel dimension for grayscale
            pred_img = pred[i, 0]
            target_img = target[i, 0]
            ssim_val = ssim(pred_img, target_img, data_range=data_range)
            ssim_vals.append(ssim_val)
        return np.mean(ssim_vals)
    elif pred.ndim == 3:  # (C, H, W)
        pred_img = pred[0]
        target_img = target[0]
        return ssim(pred_img, target_img, data_range=data_range)
    else:  # (H, W)
        return ssim(pred, target, data_range=data_range)


def compute_all_metrics(pred, target, data_range=1.0):
    """
    Compute all metrics (NMSE, PSNR, SSIM)

    Args:
        pred: Predicted image (torch.Tensor)
        target: Ground truth image (torch.Tensor)
        data_range: Range of the data (default: 1.0)

    Returns:
        Dictionary with all metrics
    """
    nmse = compute_nmse(pred, target)
    psnr = compute_psnr(pred, target, max_val=data_range)
    ssim_val = compute_ssim(pred, target, data_range=data_range)

    return {
        'nmse': nmse.item() if isinstance(nmse, torch.Tensor) else nmse,
        'psnr': psnr.item() if isinstance(psnr, torch.Tensor) else psnr,
        'ssim': ssim_val
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Create dummy data
    target = torch.rand(2, 1, 64, 64)
    pred = target + torch.randn_like(target) * 0.1

    metrics = compute_all_metrics(pred, target)
    print(f"\nMetrics:")
    print(f"  NMSE: {metrics['nmse']:.6f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")

    print("\n✓ Metrics test passed!")
