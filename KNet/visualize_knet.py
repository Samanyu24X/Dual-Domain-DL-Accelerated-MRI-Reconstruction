import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm

from model import UNetCNN
from dataset_kspace import create_kspace_data_loaders
from metrics import compute_all_metrics


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


def visualize_single_reconstruction(model, test_loader, device, save_path='knet_reconstruction.png'):
    """
    Visualize a single example showing:
    - Original (fully sampled)
    - Zero-filled (undersampled, input)
    - K-Net reconstruction
    - Difference maps
    """
    model.eval()
    
    best_example = None
    best_improvement = 0
    
    # Find a good example with visible artifacts
    print("  Searching for good example with visible artifacts...")
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        # Check each sample in batch
        for i in range(min(8, inputs.shape[0])):  # Check first 8 samples
            input_kspace = inputs[i].cpu().numpy()
            target_kspace = targets[i].cpu().numpy()
            output_kspace = outputs[i].cpu().numpy()
            
            # Convert to images
            input_img = kspace_to_image(input_kspace)
            target_img = kspace_to_image(target_kspace)
            output_img = kspace_to_image(output_kspace)
            
            # Normalize
            input_img = normalize_image(input_img)
            target_img = normalize_image(target_img)
            output_img = normalize_image(output_img)
            
            # Check for actual artifacts (input should differ from target)
            input_error = np.abs(target_img - input_img).mean()
            output_error = np.abs(target_img - output_img).mean()
            
            if input_error > 0.01:  # Has visible artifacts
                improvement = input_error - output_error
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_example = (input_kspace, target_kspace, output_kspace,
                                  input_img, target_img, output_img)
        
        if best_example is not None:
            break
    
    if best_example is None:
        print("  Warning: Could not find example with artifacts, using first sample")
        # Fallback to first sample
        for inputs, targets in test_loader:
            input_kspace = inputs[0].cpu().numpy()
            target_kspace = targets[0].cpu().numpy()
            with torch.no_grad():
                output_kspace = model(inputs.to(device))[0].cpu().numpy()
            
            input_img = normalize_image(kspace_to_image(input_kspace))
            target_img = normalize_image(kspace_to_image(target_kspace))
            output_img = normalize_image(kspace_to_image(output_kspace))
            
            best_example = (input_kspace, target_kspace, output_kspace,
                          input_img, target_img, output_img)
            break
    
    # Unpack best example
    input_kspace, target_kspace, output_kspace, input_img, target_img, output_img = best_example
    
    # Compute metrics
    input_tensor = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).float()
    target_tensor = torch.from_numpy(target_img).unsqueeze(0).unsqueeze(0).float()
    output_tensor = torch.from_numpy(output_img).unsqueeze(0).unsqueeze(0).float()
    
    metrics_input = compute_all_metrics(input_tensor, target_tensor)
    metrics_output = compute_all_metrics(output_tensor, target_tensor)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Top row: Images
    im0 = axes[0, 0].imshow(target_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth\n(Fully Sampled)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(input_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Zero-Filled Input\nPSNR: {metrics_input["psnr"]:.2f} dB\nSSIM: {metrics_input["ssim"]:.4f}', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(output_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'K-Net Reconstruction\nPSNR: {metrics_output["psnr"]:.2f} dB\nSSIM: {metrics_output["ssim"]:.4f}', 
                        fontsize=14, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Improvement
    psnr_gain = metrics_output["psnr"] - metrics_input["psnr"]
    ssim_gain = metrics_output["ssim"] - metrics_input["ssim"]
    axes[0, 3].text(0.5, 0.6, f'Improvement:\n\nPSNR: +{psnr_gain:.2f} dB\n\nSSIM: +{ssim_gain:.4f}', 
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[0, 3].set_title('Metrics Improvement', fontsize=14, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Bottom row: Difference maps
    diff_input = np.abs(target_img - input_img)
    diff_output = np.abs(target_img - output_img)
    
    im3 = axes[1, 0].imshow(target_img, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth', fontsize=12)
    axes[1, 0].axis('off')
    
    vmax_diff = max(diff_input.max(), diff_output.max())
    im4 = axes[1, 1].imshow(diff_input, cmap='hot', vmin=0, vmax=vmax_diff)
    axes[1, 1].set_title('Error Map: Zero-Filled', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    im5 = axes[1, 2].imshow(diff_output, cmap='hot', vmin=0, vmax=vmax_diff)
    axes[1, 2].set_title('Error Map: K-Net', fontsize=12, color='green')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    # Error reduction
    mean_error_input = diff_input.mean()
    mean_error_output = diff_output.mean()
    
    if mean_error_input > 0:
        error_reduction = (1 - mean_error_output / mean_error_input) * 100
    else:
        error_reduction = 0
    
    axes[1, 3].text(0.5, 0.5, f'Error Reduction:\n\n{error_reduction:.1f}%', 
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    axes[1, 3].set_title('Mean Error Reduction', fontsize=14, fontweight='bold')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved reconstruction visualization to {save_path}")
    print(f"  Selected example with {error_reduction:.1f}% error reduction")
    plt.close()


def visualize_kspace_comparison(model, test_loader, device, save_path='kspace_comparison.png'):
    """
    Visualize k-space: input (undersampled) vs output (reconstructed)
    """
    model.eval()
    
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        # Take first sample and convert to numpy
        input_kspace = inputs[0].cpu().numpy()
        target_kspace = targets[0].cpu().numpy()
        output_kspace = outputs[0].cpu().numpy()
        
        # Compute magnitude of k-space (for visualization)
        input_mag = np.sqrt(input_kspace[0]**2 + input_kspace[1]**2)
        target_mag = np.sqrt(target_kspace[0]**2 + target_kspace[1]**2)
        output_mag = np.sqrt(output_kspace[0]**2 + output_kspace[1]**2)
        
        # Log scale for better visualization
        input_mag_log = np.log(input_mag + 1e-10)
        target_mag_log = np.log(target_mag + 1e-10)
        output_mag_log = np.log(output_mag + 1e-10)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Top row: K-space magnitude (log scale)
        vmin = min(input_mag_log.min(), target_mag_log.min(), output_mag_log.min())
        vmax = max(input_mag_log.max(), target_mag_log.max(), output_mag_log.max())
        
        im0 = axes[0, 0].imshow(target_mag_log, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Target K-Space\n(Fully Sampled)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, label='Log Magnitude')
        
        im1 = axes[0, 1].imshow(input_mag_log, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Input K-Space\n(Undersampled)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='Log Magnitude')
        
        im2 = axes[0, 2].imshow(output_mag_log, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('K-Net Output K-Space\n(Reconstructed)', fontsize=14, fontweight='bold', color='green')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, label='Log Magnitude')
        
        # Bottom row: Difference maps in k-space
        diff_input = np.abs(target_mag - input_mag)
        diff_output = np.abs(target_mag - output_mag)
        
        axes[1, 0].imshow(target_mag_log, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Target K-Space', fontsize=12)
        axes[1, 0].axis('off')
        
        vmax_diff = max(diff_input.max(), diff_output.max())
        im3 = axes[1, 1].imshow(diff_input, cmap='hot', vmin=0, vmax=vmax_diff)
        axes[1, 1].set_title('K-Space Error: Input', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
        
        im4 = axes[1, 2].imshow(diff_output, cmap='hot', vmin=0, vmax=vmax_diff)
        axes[1, 2].set_title('K-Space Error: K-Net Output', fontsize=12, color='green')
        axes[1, 2].axis('off')
        plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved k-space comparison to {save_path}")
        plt.close()
        
        break


def evaluate_test_set(model, test_loader, device):
    """Evaluate model on entire test set and return metrics"""
    model.eval()
    
    all_psnr = []
    all_ssim = []
    all_nmse = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating test set'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # Convert batch to images
            output_imgs = []
            target_imgs = []
            
            for i in range(outputs.shape[0]):
                # Get single sample k-space
                out_k = outputs[i].cpu()
                tar_k = targets[i].cpu()
                
                # Convert to image
                out_img = kspace_to_image(out_k.unsqueeze(0))  # Add batch dim
                tar_img = kspace_to_image(tar_k.unsqueeze(0))
                
                # Handle tensor output
                if isinstance(out_img, torch.Tensor):
                    out_img = out_img.squeeze().numpy()
                if isinstance(tar_img, torch.Tensor):
                    tar_img = tar_img.squeeze().numpy()
                
                # Normalize
                out_img = normalize_image(out_img)
                tar_img = normalize_image(tar_img)
                
                output_imgs.append(out_img)
                target_imgs.append(tar_img)
            
            # Convert to proper tensor format [B, 1, H, W]
            output_batch = torch.from_numpy(np.array(output_imgs)).unsqueeze(1).float()
            target_batch = torch.from_numpy(np.array(target_imgs)).unsqueeze(1).float()
            
            # Compute metrics for batch
            metrics = compute_all_metrics(output_batch, target_batch)
            
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
            all_nmse.append(metrics['nmse'])
    
    return {
        'psnr': np.array(all_psnr),
        'ssim': np.array(all_ssim),
        'nmse': np.array(all_nmse)
    }


def plot_metrics_distribution(metrics, save_path='knet_metrics_distribution.png'):
    """Plot distribution of metrics across test set"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PSNR
    axes[0].hist(metrics['psnr'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(metrics['psnr'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["psnr"].mean():.2f} dB')
    axes[0].set_xlabel('PSNR (dB)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('PSNR Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # SSIM
    axes[1].hist(metrics['ssim'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(metrics['ssim'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["ssim"].mean():.4f}')
    axes[1].set_xlabel('SSIM', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('SSIM Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # NMSE
    axes[2].hist(metrics['nmse'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[2].axvline(metrics['nmse'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["nmse"].mean():.6f}')
    axes[2].set_xlabel('NMSE', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[2].set_title('NMSE Distribution', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics distribution to {save_path}")
    plt.close()


def visualize_multiple_examples(model, test_loader, device, num_examples=6, save_path='knet_multiple_examples.png'):
    """Visualize multiple reconstruction examples in a grid"""
    model.eval()
    
    examples = []
    count = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            for i in range(inputs.shape[0]):
                if count >= num_examples:
                    break
                
                input_kspace = inputs[i].cpu().numpy()
                target_kspace = targets[i].cpu().numpy()
                output_kspace = outputs[i].cpu().numpy()
                
                input_img = normalize_image(kspace_to_image(input_kspace))
                target_img = normalize_image(kspace_to_image(target_kspace))
                output_img = normalize_image(kspace_to_image(output_kspace))
                
                examples.append((target_img, input_img, output_img))
                count += 1
            
            if count >= num_examples:
                break
    
    # Create grid
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
    
    for idx, (target, input_img, output) in enumerate(examples):
        axes[idx, 0].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[idx, 0].set_title('Ground Truth' if idx == 0 else '', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(input_img, cmap='gray', vmin=0, vmax=1)
        axes[idx, 1].set_title('Zero-Filled Input' if idx == 0 else '', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(output, cmap='gray', vmin=0, vmax=1)
        axes[idx, 2].set_title('K-Net Reconstruction' if idx == 0 else '', fontsize=12, fontweight='bold', color='green')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved multiple examples to {save_path}")
    plt.close()


def main():
    """Main visualization pipeline"""
    print("K-Net Visualization Script")
    print("=" * 50)
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    _, _, test_loader = create_kspace_data_loaders(
        data_dir='data',
        batch_size=8,
        num_workers=0
    )
    
    # Load model
    print("Loading K-Net model...")
    model = UNetCNN(in_channels=2, out_channels=2, base_channels=64)
    checkpoint = torch.load('checkpoints/knet_fixed/best_knet.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation metrics: PSNR={checkpoint['val_metrics']['psnr']:.2f} dB, "
          f"SSIM={checkpoint['val_metrics']['ssim']:.4f}\n")
    
    # Create visualizations
    print("Creating visualizations...")
    
    print("\n1. Single reconstruction example...")
    visualize_single_reconstruction(model, test_loader, device, 'knet_reconstruction.png')
    
    print("\n2. K-space comparison...")
    visualize_kspace_comparison(model, test_loader, device, 'knet_kspace_comparison.png')
    
    print("\n3. Multiple examples grid...")
    visualize_multiple_examples(model, test_loader, device, num_examples=6, save_path='knet_multiple_examples.png')
    
    print("\n4. Evaluating on test set...")
    test_metrics = evaluate_test_set(model, test_loader, device)
    
    print(f"\nTest Set Results:")
    print(f"  PSNR: {test_metrics['psnr'].mean():.2f} ± {test_metrics['psnr'].std():.2f} dB")
    print(f"  SSIM: {test_metrics['ssim'].mean():.4f} ± {test_metrics['ssim'].std():.4f}")
    print(f"  NMSE: {test_metrics['nmse'].mean():.6f} ± {test_metrics['nmse'].std():.6f}")
    
    print("\n5. Plotting metrics distributions...")
    plot_metrics_distribution(test_metrics, 'knet_metrics_distribution.png')
    
    print("\n" + "=" * 50)
    print("All visualizations complete!")
    print("Generated files:")
    print("  - knet_reconstruction.png")
    print("  - knet_kspace_comparison.png")
    print("  - knet_multiple_examples.png")
    print("  - knet_metrics_distribution.png")


if __name__ == "__main__":
    main()