import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

from model import UNetCNN
from dataset_kspace import create_kspace_data_loaders
from metrics import compute_all_metrics


def kspace_to_image(kspace_tensor):
    """
    Convert k-space tensor to image domain
    
    Args:
        kspace_tensor: [B, 2, H, W] where channel 0=real, 1=imaginary
    
    Returns:
        image_tensor: [B, 1, H, W] magnitude image
    """
    # Combine real and imaginary into complex
    kspace_complex = torch.complex(kspace_tensor[:, 0], kspace_tensor[:, 1])
    
    # Inverse FFT with proper shifts
    image_complex = torch.fft.ifftshift(kspace_complex, dim=(-2, -1))
    image_complex = torch.fft.ifft2(image_complex, norm='ortho')  # Use orthonormal scaling
    image_complex = torch.fft.fftshift(image_complex, dim=(-2, -1))
    
    # Take magnitude
    image = torch.abs(image_complex).unsqueeze(1)
    
    return image


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss - weighted combination of k-space and image domain
        loss_kspace = criterion(outputs, targets)
        
        # Also compute loss in image domain for stability
        output_img = kspace_to_image(outputs)
        target_img = kspace_to_image(targets)
        
        # Normalize images
        output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min() + 1e-8)
        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
        
        loss_image = criterion(output_img, target_img)
        
        # Combined loss: 80% k-space, 20% image
        loss = 0.8 * loss_kspace + 0.2 * loss_image

        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_nmse = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Loss in k-space
            loss = criterion(outputs, targets)

            # Convert to image domain for metrics
            output_images = kspace_to_image(outputs)
            target_images = kspace_to_image(targets)

            # Normalize to [0, 1]
            output_images = (output_images - output_images.min()) / (output_images.max() - output_images.min() + 1e-8)
            target_images = (target_images - target_images.min()) / (target_images.max() - target_images.min() + 1e-8)

            # Compute metrics
            metrics = compute_all_metrics(output_images, target_images)

            running_loss += loss.item() * inputs.size(0)
            running_nmse += metrics['nmse'] * inputs.size(0)
            running_psnr += metrics['psnr'] * inputs.size(0)
            running_ssim += metrics['ssim'] * inputs.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{metrics["psnr"]:.2f}',
                'ssim': f'{metrics["ssim"]:.4f}'
            })

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_nmse = running_nmse / len(val_loader.dataset)
    epoch_psnr = running_psnr / len(val_loader.dataset)
    epoch_ssim = running_ssim / len(val_loader.dataset)

    return {
        'loss': epoch_loss,
        'nmse': epoch_nmse,
        'psnr': epoch_psnr,
        'ssim': epoch_ssim
    }


def train_model(args):
    """Main training function"""

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create data loaders
    print('\nLoading k-space data...')
    train_loader, val_loader, test_loader = create_kspace_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    print(f'\nCreating K-Net (U-Net for k-space)')
    model = UNetCNN(
        in_channels=2,
        out_channels=2,
        base_channels=args.base_channels
    )

    model = model.to(device)
    print(f'Model parameters: {model.num_parameters():,}')

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer - slightly lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_psnr = 0.0  # Track PSNR instead of loss
    best_metrics = {}

    print(f'\nStarting K-Net training for {args.epochs} epochs...\n')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        scheduler.step(val_metrics['loss'])

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('NMSE/val', val_metrics['nmse'], epoch)
        writer.add_scalar('PSNR/val', val_metrics['psnr'], epoch)
        writer.add_scalar('SSIM/val', val_metrics['ssim'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch {epoch}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss:   {val_metrics["loss"]:.4f}')
        print(f'  Val NMSE:   {val_metrics["nmse"]:.6f}')
        print(f'  Val PSNR:   {val_metrics["psnr"]:.2f} dB')
        print(f'  Val SSIM:   {val_metrics["ssim"]:.4f}')
        print(f'  LR:         {optimizer.param_groups[0]["lr"]:.6f}\n')

        # Save best model based on PSNR
        if val_metrics['psnr'] > best_val_psnr:
            best_val_psnr = val_metrics['psnr']
            best_metrics = val_metrics.copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_dir / 'best_knet.pth')
            print(f'Saved best K-Net (PSNR: {val_metrics["psnr"]:.2f} dB)\n')

        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_dir / f'knet_epoch_{epoch}.pth')

    print(f'\nK-Net training complete!')
    print(f'Best Validation Metrics:')
    print(f'  PSNR: {best_metrics.get("psnr", 0):.2f} dB')
    print(f'  SSIM: {best_metrics.get("ssim", 0):.4f}')
    print(f'  NMSE: {best_metrics.get("nmse", 0):.6f}')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train K-Net for MRI Reconstruction')

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)  # Start with 5 for testing
    parser.add_argument('--lr', type=float, default=5e-5)  # Lower learning rate
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--log_dir', type=str, default='logs/knet_fixed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/knet_fixed')
    parser.add_argument('--save_freq', type=int, default=5)

    args = parser.parse_args()
    train_model(args)