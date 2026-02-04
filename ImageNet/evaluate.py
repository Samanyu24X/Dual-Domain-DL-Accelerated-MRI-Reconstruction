"""
Evaluation script to compute metrics on validation and test sets
"""
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from model import ImageDomainCNN, UNetCNN
from dataset import create_data_loaders
from metrics import compute_all_metrics


def evaluate_model(model, data_loader, device, split_name='Test'):
    """
    Evaluate model on a dataset

    Args:
        model: Trained model
        data_loader: DataLoader for the dataset
        device: Device to run on
        split_name: Name of the split (e.g., 'Val', 'Test')

    Returns:
        Dictionary with averaged metrics
    """
    model.eval()

    running_nmse = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_samples = 0

    all_nmse = []
    all_psnr = []
    all_ssim = []

    print(f'\nEvaluating on {split_name} set...')

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'{split_name} Evaluation')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute metrics for each sample in batch
            for i in range(inputs.size(0)):
                metrics = compute_all_metrics(
                    outputs[i:i+1],
                    targets[i:i+1]
                )

                all_nmse.append(metrics['nmse'])
                all_psnr.append(metrics['psnr'])
                all_ssim.append(metrics['ssim'])

                running_nmse += metrics['nmse']
                running_psnr += metrics['psnr']
                running_ssim += metrics['ssim']
                num_samples += 1

            pbar.set_postfix({
                'psnr': f'{np.mean(all_psnr):.2f}',
                'ssim': f'{np.mean(all_ssim):.4f}'
            })

    # Compute statistics
    avg_metrics = {
        'nmse': {
            'mean': np.mean(all_nmse),
            'std': np.std(all_nmse),
            'min': np.min(all_nmse),
            'max': np.max(all_nmse)
        },
        'psnr': {
            'mean': np.mean(all_psnr),
            'std': np.std(all_psnr),
            'min': np.min(all_psnr),
            'max': np.max(all_psnr)
        },
        'ssim': {
            'mean': np.mean(all_ssim),
            'std': np.std(all_ssim),
            'min': np.min(all_ssim),
            'max': np.max(all_ssim)
        }
    }

    return avg_metrics


def print_metrics(metrics, split_name='Test'):
    """Print metrics in a nice format"""
    print(f'\n{"=" * 60}')
    print(f'{split_name} Set Results')
    print(f'{"=" * 60}')

    print(f'\nNMSE (Normalized Mean Squared Error):')
    print(f'  Mean: {metrics["nmse"]["mean"]:.6f} ± {metrics["nmse"]["std"]:.6f}')
    print(f'  Range: [{metrics["nmse"]["min"]:.6f}, {metrics["nmse"]["max"]:.6f}]')

    print(f'\nPSNR (Peak Signal-to-Noise Ratio):')
    print(f'  Mean: {metrics["psnr"]["mean"]:.2f} ± {metrics["psnr"]["std"]:.2f} dB')
    print(f'  Range: [{metrics["psnr"]["min"]:.2f}, {metrics["psnr"]["max"]:.2f}] dB')

    print(f'\nSSIM (Structural Similarity Index):')
    print(f'  Mean: {metrics["ssim"]["mean"]:.4f} ± {metrics["ssim"]["std"]:.4f}')
    print(f'  Range: [{metrics["ssim"]["min"]:.4f}, {metrics["ssim"]["max"]:.4f}]')

    print(f'{"=" * 60}\n')


def main(args):
    """Main evaluation function"""

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

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f'\nLoading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')

    # Evaluate on validation set
    if args.eval_val:
        val_metrics = evaluate_model(model, val_loader, device, split_name='Validation')
        print_metrics(val_metrics, split_name='Validation')

        # Save results
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'val_metrics.json', 'w') as f:
                json.dump(val_metrics, f, indent=2)
            print(f'Saved validation metrics to {output_dir / "val_metrics.json"}')

    # Evaluate on test set
    if args.eval_test:
        test_metrics = evaluate_model(model, test_loader, device, split_name='Test')
        print_metrics(test_metrics, split_name='Test')

        # Save results
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'test_metrics.json', 'w') as f:
                json.dump(test_metrics, f, indent=2)
            print(f'Saved test metrics to {output_dir / "test_metrics.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MRI Reconstruction Model')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Root data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # Model parameters
    parser.add_argument('--model', type=str, default='imagecnn', choices=['imagecnn', 'unet'],
                        help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels')
    parser.add_argument('--num_res_blocks', type=int, default=5, help='Number of residual blocks')

    # Evaluation parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--eval_val', action='store_true', help='Evaluate on validation set')
    parser.add_argument('--eval_test', action='store_true', help='Evaluate on test set')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')

    args = parser.parse_args()

    # If neither flag is set, evaluate on both
    if not args.eval_val and not args.eval_test:
        args.eval_val = True
        args.eval_test = True

    main(args)
