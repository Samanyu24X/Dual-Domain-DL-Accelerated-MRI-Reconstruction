"""
Training script for Image-Domain CNN
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

from model import ImageDomainCNN, UNetCNN
from dataset import create_data_loaders
from metrics import compute_all_metrics


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
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

            # Compute loss
            loss = criterion(outputs, targets)

            # Compute all metrics
            metrics = compute_all_metrics(outputs, targets)

            # Statistics
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create data loaders
    print('\nLoading data...')
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    print(f'\nCreating model: {args.model}')
    if args.model == 'imagecnn':
        model = ImageDomainCNN(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
            num_res_blocks=args.num_res_blocks
        )
    elif args.model == 'unet':
        model = UNetCNN(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    print(f'Model parameters: {model.num_parameters():,}')

    # Loss function
    if args.loss == 'l1':
        criterion = nn.L1Loss()
    elif args.loss == 'l2':
        criterion = nn.MSELoss()
    elif args.loss == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    best_metrics = {}

    print(f'\nStarting training for {args.epochs} epochs...\n')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Learning rate scheduling
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

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_metrics = val_metrics.copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_dir / 'best_model.pth')
            print(f'  ✓ Saved best model (Loss: {val_metrics["loss"]:.4f}, PSNR: {val_metrics["psnr"]:.2f} dB, SSIM: {val_metrics["ssim"]:.4f})\n')

        # Save checkpoint every N epochs
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')

    print(f'\nTraining complete!')
    print(f'Best Validation Metrics:')
    print(f'  Loss: {best_val_loss:.4f}')
    print(f'  NMSE: {best_metrics.get("nmse", 0):.6f}')
    print(f'  PSNR: {best_metrics.get("psnr", 0):.2f} dB')
    print(f'  SSIM: {best_metrics.get("ssim", 0):.4f}')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image-Domain CNN for MRI Reconstruction')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Root data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # Model parameters
    parser.add_argument('--model', type=str, default='imagecnn', choices=['imagecnn', 'unet'],
                        help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels')
    parser.add_argument('--num_res_blocks', type=int, default=5, help='Number of residual blocks')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2', 'smooth_l1'],
                        help='Loss function')

    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N epochs')

    args = parser.parse_args()

    train_model(args)
