"""
Test K-Net robustness to noise by adding noise to k-space input
and comparing reconstruction to ground truth
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

from model import UNetCNN
from dataset_kspace import create_kspace_data_loaders


def kspace_to_image(kspace_tensor):
    """Convert k-space tensor to image domain"""
    if isinstance(kspace_tensor, torch.Tensor):
        kspace_complex = torch.complex(kspace_tensor[:, 0], kspace_tensor[:, 1])
        image_complex = torch.fft.ifftshift(kspace_complex, dim=(-2, -1))
        image_complex = torch.fft.ifft2(image_complex, norm='ortho')
        image_complex = torch.fft.fftshift(image_complex, dim=(-2, -1))
        image = torch.abs(image_complex)
    else:
        kspace_complex = kspace_tensor[0] + 1j * kspace_tensor[1]
        image_complex = np.fft.ifftshift(kspace_complex)
        image_complex = np.fft.ifft2(image_complex, norm='ortho')
        image_complex = np.fft.fftshift(image_complex)
        image = np.abs(image_complex)
    return image


def normalize_image(img):
    """Normalize image to [0, 1]"""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return img


def add_noise_to_kspace(kspace, noise_level=0.1):
    """
    Add Gaussian noise to k-space data
    
    Args:
        kspace: k-space tensor [2, H, W] (real, imag)
        noise_level: standard deviation of noise relative to signal
    
    Returns:
        noisy_kspace: k-space with added noise
    """
    # Compute signal magnitude
    signal_magnitude = torch.sqrt(kspace[0]**2 + kspace[1]**2).mean()
    
    # Add noise to both real and imaginary channels
    noise_real = torch.randn_like(kspace[0]) * noise_level * signal_magnitude
    noise_imag = torch.randn_like(kspace[1]) * noise_level * signal_magnitude
    
    noisy_kspace = kspace.clone()
    noisy_kspace[0] += noise_real
    noisy_kspace[1] += noise_imag
    
    return noisy_kspace


def test_noise_robustness():
    """Test K-Net reconstruction with noisy k-space input"""
    
    print("=" * 80)
    print("K-NET NOISE ROBUSTNESS TEST")
    print("=" * 80)
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading K-Net model...")
    model = UNetCNN(in_channels=2, out_channels=2, base_channels=64)
    checkpoint = torch.load('checkpoints/knet_fixed/best_knet.pth', 
                           map_location=device, 
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_loader = create_kspace_data_loaders(
        data_dir='data',
        batch_size=1,
        num_workers=0
    )
    
    # Get one example
    for inputs, targets in test_loader:
        input_kspace = inputs[0]  # [2, H, W]
        target_kspace = targets[0]
        break
    
    # Test different noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    
    # Create output directory
    output_dir = Path('knet_noise_test')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nTesting noise levels: {noise_levels}")
    print("Generating reconstructions...")
    
    results = []
    
    for noise_level in noise_levels:
        print(f"\n  Noise level: {noise_level:.2f}")
        
        # Add noise to input
        if noise_level > 0:
            noisy_input = add_noise_to_kspace(input_kspace, noise_level)
        else:
            noisy_input = input_kspace.clone()
        
        # Convert to images (before reconstruction)
        noisy_image = normalize_image(kspace_to_image(noisy_input.numpy()))
        target_image = normalize_image(kspace_to_image(target_kspace.numpy()))
        
        # Reconstruct with K-Net
        with torch.no_grad():
            noisy_input_batch = noisy_input.unsqueeze(0).to(device)
            reconstructed_kspace = model(noisy_input_batch)[0].cpu()
        
        # Convert reconstruction to image
        reconstructed_image = normalize_image(kspace_to_image(reconstructed_kspace.numpy()))
        
        # Compute metrics
        mse_noisy = np.mean((target_image - noisy_image) ** 2)
        mse_recon = np.mean((target_image - reconstructed_image) ** 2)
        
        psnr_noisy = 20 * np.log10(1.0 / np.sqrt(mse_noisy)) if mse_noisy > 0 else 100
        psnr_recon = 20 * np.log10(1.0 / np.sqrt(mse_recon)) if mse_recon > 0 else 100
        
        print(f"    Noisy input PSNR: {psnr_noisy:.2f} dB")
        print(f"    Reconstructed PSNR: {psnr_recon:.2f} dB")
        print(f"    Improvement: {psnr_recon - psnr_noisy:.2f} dB")
        
        results.append({
            'noise_level': noise_level,
            'noisy_image': noisy_image,
            'reconstructed_image': reconstructed_image,
            'target_image': target_image,
            'psnr_noisy': psnr_noisy,
            'psnr_recon': psnr_recon
        })
    
    # Create comprehensive visualization
    print("\nCreating visualizations...")
    
    # Figure 1: Grid showing all noise levels
    fig, axes = plt.subplots(3, len(noise_levels), figsize=(20, 12))
    
    for i, result in enumerate(results):
        # Top row: Noisy input
        axes[0, i].imshow(result['noisy_image'], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Noisy Input\nNoise: {result["noise_level"]:.2f}\nPSNR: {result["psnr_noisy"]:.1f} dB',
                            fontsize=10)
        axes[0, i].axis('off')
        
        # Middle row: K-Net reconstruction
        axes[1, i].imshow(result['reconstructed_image'], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'K-Net Output\nPSNR: {result["psnr_recon"]:.1f} dB',
                            fontsize=10, color='green', fontweight='bold')
        axes[1, i].axis('off')
        
        # Bottom row: Ground truth
        axes[2, i].imshow(result['target_image'], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title('Ground Truth', fontsize=10)
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Noisy Input', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('K-Net Reconstruction', fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel('Ground Truth', fontsize=14, fontweight='bold')
    
    plt.suptitle('K-Net Noise Robustness Test', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'noise_robustness_grid.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'noise_robustness_grid.png'}")
    plt.close()
    
    # Figure 2: PSNR comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    noise_vals = [r['noise_level'] for r in results]
    psnr_noisy_vals = [r['psnr_noisy'] for r in results]
    psnr_recon_vals = [r['psnr_recon'] for r in results]

    ax.plot(noise_vals, psnr_noisy_vals, 'o-', label='Noisy Input', linewidth=2, markersize=8)
    ax.plot(noise_vals, psnr_recon_vals, 's-', label='K-Net Reconstruction', 
            linewidth=2, markersize=8, color='green')

    ax.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('K-Net Denoising Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, None)  # ← ADD THIS LINE: Set y-axis to start at 0
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'psnr_vs_noise.png', dpi=300, bbox_inches='tight')
    
    # Figure 3: Single example comparison (highest noise)
    highest_noise = results[-1]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(highest_noise['target_image'], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(highest_noise['noisy_image'], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Noisy Input (o={highest_noise["noise_level"]:.2f})\n'
                     f'PSNR: {highest_noise["psnr_noisy"]:.2f} dB', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(highest_noise['reconstructed_image'], cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'K-Net Reconstruction\nPSNR: {highest_noise["psnr_recon"]:.2f} dB', 
                     fontsize=14, fontweight='bold', color='green')
    axes[2].axis('off')
    
    # Error map
    error_map = np.abs(highest_noise['target_image'] - highest_noise['reconstructed_image'])
    im = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=0.2)
    axes[3].set_title('Reconstruction Error', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    
    plt.suptitle(f'K-Net Denoising Example (Noise Level: {highest_noise["noise_level"]:.2f})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'single_example_denoising.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'single_example_denoising.png'}")
    plt.close()
    
    # Save summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary_file = output_dir / 'noise_test_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("K-Net Noise Robustness Test Results\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            improvement = result['psnr_recon'] - result['psnr_noisy']
            f.write(f"Noise Level: {result['noise_level']:.2f}\n")
            f.write(f"  Noisy Input PSNR:  {result['psnr_noisy']:>6.2f} dB\n")
            f.write(f"  Reconstructed PSNR: {result['psnr_recon']:>6.2f} dB\n")
            f.write(f"  Improvement:        {improvement:>6.2f} dB\n")
            f.write("\n")
            
            print(f"\nNoise Level: {result['noise_level']:.2f}")
            print(f"  Improvement: {improvement:>6.2f} dB")
    
    print(f"\n  Saved summary: {summary_file}")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files in '{output_dir}/':")
    print("  - noise_robustness_grid.png (all noise levels)")
    print("  - psnr_vs_noise.png (performance curve)")
    print("  - single_example_denoising.png (detailed example)")
    print("  - noise_test_summary.txt (numerical results)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_noise_robustness()