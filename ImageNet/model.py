"""
Image-Domain CNN Architecture for MRI Reconstruction
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with Conv2D -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += residual
        return nn.functional.relu(out)


class ImageDomainCNN(nn.Module):
    """
    Image-Domain CNN for MRI Reconstruction

    Architecture:
        - Initial feature extraction
        - Multiple residual blocks
        - Feature refinement
        - Output reconstruction

    Args:
        in_channels: Number of input channels (default: 1 for grayscale MRI)
        out_channels: Number of output channels (default: 1)
        base_channels: Number of base feature channels (default: 64)
        num_res_blocks: Number of residual blocks (default: 5)
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_res_blocks=5):
        super(ImageDomainCNN, self).__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Encoder path - increase channels, decrease spatial dimensions
        self.encoder1 = ConvBlock(base_channels, base_channels * 2, stride=2)  # /2
        self.encoder2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)  # /4

        # Residual blocks at bottleneck
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels * 4) for _ in range(num_res_blocks)]
        )

        # Decoder path - decrease channels, increase spatial dimensions
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Final reconstruction
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.initial(x)

        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)

        # Residual blocks
        x2 = self.res_blocks(x2)

        # Decoder
        x = self.decoder2(x2)
        x = self.decoder1(x)

        # Output
        x = self.output(x)

        return x

    def num_parameters(self):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetCNN(nn.Module):
    """
    U-Net style CNN for MRI Reconstruction with skip connections

    Args:
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 1)
        base_channels: Number of base feature channels (default: 64)
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(UNetCNN, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Output
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.output(dec1)

    def num_parameters(self):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_models():
    """Test function to verify model architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test ImageDomainCNN
    print("Testing ImageDomainCNN...")
    model1 = ImageDomainCNN(in_channels=1, out_channels=1, base_channels=64, num_res_blocks=5).to(device)
    x = torch.randn(2, 1, 256, 256).to(device)  # Batch of 2 images
    y = model1(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Number of parameters: {model1.num_parameters():,}")

    # Test UNetCNN
    print("\nTesting UNetCNN...")
    model2 = UNetCNN(in_channels=1, out_channels=1, base_channels=64).to(device)
    y2 = model2(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y2.shape}")
    print(f"  Number of parameters: {model2.num_parameters():,}")


if __name__ == "__main__":
    test_models()
