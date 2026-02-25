import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Tuple


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class UNetBlock(nn.Module):
    """U-Net building block with skip connections"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down: bool = True,
        use_dropout: bool = False,
        use_norm: bool = True
    ):
        super().__init__()

        self.down = down

        if down:
            # Encoder block: Conv -> Norm -> LeakyReLU
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if use_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            # Decoder block: ConvTranspose -> Norm -> ReLU
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if use_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttributeInjector(nn.Module):
    """Inject attribute vector into feature maps using adaptive normalization"""

    def __init__(self, feature_dim: int, attr_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.attr_dim = attr_dim

        # Project attributes to feature dimension
        self.attr_proj = nn.Sequential(
            nn.Linear(attr_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim * 2)  # For scale and bias
        )

    def forward(self, features: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
            attributes: [B, attr_dim]
        Returns:
            modulated_features: [B, C, H, W]
        """
        B, C, H, W = features.shape

        # Project attributes
        attr_params = self.attr_proj(attributes)  # [B, C*2]
        scale, bias = torch.chunk(attr_params, 2, dim=1)  # Each [B, C]

        # Reshape for broadcasting
        scale = scale.view(B, C, 1, 1)  # [B, C, 1, 1]
        bias = bias.view(B, C, 1, 1)    # [B, C, 1, 1]

        # Apply adaptive normalization
        # Normalize features (skip for 1x1 spatial - normalization is meaningless for single element)
        if H > 1 and W > 1:
            features_norm = F.instance_norm(features)
        else:
            features_norm = features

        # Apply learned scale and bias
        modulated_features = features_norm * (1 + scale) + bias

        return modulated_features


class Generator(nn.Module):
    """
    U-Net Generator with attribute injection at bottleneck.

    Takes edge maps and attribute vectors as input, outputs RGB images.
    """

    def __init__(self, attr_dim: int, ngf: int = 64):
        """
        Args:
            attr_dim: Dimension of attribute vector
            ngf: Number of generator filters in first conv layer
        """
        super().__init__()

        self.attr_dim = attr_dim

        # Encoder (downsampling)
        self.e1 = nn.Conv2d(3, ngf, 4, 2, 1)  # No norm for first layer
        self.e2 = UNetBlock(ngf, ngf * 2, down=True, use_norm=True)
        self.e3 = UNetBlock(ngf * 2, ngf * 4, down=True, use_norm=True)
        self.e4 = UNetBlock(ngf * 4, ngf * 8, down=True, use_norm=True)
        self.e5 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=True)
        self.e6 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=True)
        self.e7 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=True)

        # Bottleneck
        self.bottleneck = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=False)

        # Attribute injection at bottleneck
        self.attr_injector = AttributeInjector(ngf * 8, attr_dim)

        # Decoder (upsampling)
        self.d1 = UNetBlock(ngf * 8, ngf * 8, down=False, use_dropout=True)
        self.d2 = UNetBlock(ngf * 16, ngf * 8, down=False, use_dropout=True)  # +skip
        self.d3 = UNetBlock(ngf * 16, ngf * 8, down=False, use_dropout=True)  # +skip
        self.d4 = UNetBlock(ngf * 16, ngf * 8, down=False, use_dropout=False)  # +skip
        self.d5 = UNetBlock(ngf * 16, ngf * 4, down=False, use_dropout=False)  # +skip
        self.d6 = UNetBlock(ngf * 8, ngf * 2, down=False, use_dropout=False)   # +skip
        self.d7 = UNetBlock(ngf * 4, ngf, down=False, use_dropout=False)       # +skip

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1),  # +skip
            nn.Tanh()
        )

        # Initialize weights
        self.apply(weights_init)

    def forward(self, edges: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edges: Edge maps [B, 3, H, W]
            attributes: Attribute vectors [B, attr_dim]

        Returns:
            generated_images: [B, 3, H, W]
        """
        # Encoder
        e1 = F.leaky_relu(self.e1(edges), 0.2)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        # Bottleneck
        bottleneck = self.bottleneck(e7)

        # Inject attributes
        bottleneck_with_attr = self.attr_injector(bottleneck, attributes)

        # Decoder with skip connections
        d1 = self.d1(bottleneck_with_attr)
        d2 = self.d2(torch.cat([d1, e7], dim=1))  # Skip connection
        d3 = self.d3(torch.cat([d2, e6], dim=1))  # Skip connection
        d4 = self.d4(torch.cat([d3, e5], dim=1))  # Skip connection
        d5 = self.d5(torch.cat([d4, e4], dim=1))  # Skip connection
        d6 = self.d6(torch.cat([d5, e3], dim=1))  # Skip connection
        d7 = self.d7(torch.cat([d6, e2], dim=1))  # Skip connection

        # Final output
        output = self.final(torch.cat([d7, e1], dim=1))  # Skip connection

        return output


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with spectral normalization.

    Conditioned on edge maps, attributes, and images.
    """

    def __init__(self, attr_dim: int, ndf: int = 64):
        """
        Args:
            attr_dim: Dimension of attribute vector
            ndf: Number of discriminator filters in first conv layer
        """
        super().__init__()

        self.attr_dim = attr_dim

        # Project attributes to spatial maps
        self.attr_proj = nn.Sequential(
            nn.Linear(attr_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256 * 256),  # Will reshape to [B, 1, 256, 256]
        )

        # Input: edges (3) + image (3) + attr_map (1) = 7 channels
        self.conv1 = spectral_norm(nn.Conv2d(7, ndf, 4, 2, 1))

        self.conv2 = spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1))
        self.norm2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1))
        self.norm3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1))
        self.norm4 = nn.BatchNorm2d(ndf * 8)

        # Final patch output
        self.conv5 = spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1))

        # Initialize weights
        self.apply(weights_init)

    def forward(
        self,
        edges: torch.Tensor,
        attributes: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            edges: Edge maps [B, 3, H, W]
            attributes: Attribute vectors [B, attr_dim]
            images: RGB images [B, 3, H, W]

        Returns:
            patch_outputs: [B, 1, H_patch, W_patch]
        """
        B, _, H, W = images.shape

        # Project attributes to spatial maps
        attr_spatial = self.attr_proj(attributes)  # [B, H*W]
        attr_map = attr_spatial.view(B, 1, H, W)   # [B, 1, H, W]

        # Concatenate all inputs
        x = torch.cat([edges, images, attr_map], dim=1)  # [B, 7, H, W]

        # Apply discriminator layers
        x = F.leaky_relu(self.conv1(x), 0.2)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv4(x)
        x = self.norm4(x)
        x = F.leaky_relu(x, 0.2)

        # Final patch prediction
        output = self.conv5(x)

        return output


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    """

    def __init__(self, feature_layers: list = None, device='cuda'):
        super().__init__()

        if feature_layers is None:
            feature_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

        self.feature_layers = feature_layers
        self.device = device

        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg.to(device)
        self.vgg.eval()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # VGG normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Layer name mapping
        self.layer_name_mapping = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted images [-1, 1]
            target: Target images [-1, 1]

        Returns:
            perceptual_loss: Scalar loss
        """
        # Convert from [-1, 1] to [0, 1]
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0

        # Normalize for VGG
        pred = self.normalize(pred)
        target = self.normalize(target)

        # Extract features
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        # Compute loss
        loss = 0.0
        for layer in self.feature_layers:
            if layer in pred_features and layer in target_features:
                loss += F.mse_loss(pred_features[layer], target_features[layer])

        return loss

    def _extract_features(self, x: torch.Tensor) -> dict:
        """Extract features from specified layers"""
        features = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.feature_layers:
                    features[layer_name] = x

        return features


def hinge_loss_generator(fake_output: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator"""
    return -torch.mean(fake_output)


def hinge_loss_discriminator(
    real_output: torch.Tensor,
    fake_output: torch.Tensor
) -> torch.Tensor:
    """Hinge loss for discriminator"""
    real_loss = torch.mean(F.relu(1.0 - real_output))
    fake_loss = torch.mean(F.relu(1.0 + fake_output))
    return real_loss + fake_loss


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing models on {device}")

    # Parameters
    batch_size = 4
    attr_dim = 100
    image_size = 256

    # Create dummy inputs
    edges = torch.randn(batch_size, 3, image_size, image_size).to(device)
    attributes = torch.randn(batch_size, attr_dim).to(device)
    images = torch.randn(batch_size, 3, image_size, image_size).to(device)

    # Test Generator
    print("Testing Generator...")
    generator = Generator(attr_dim=attr_dim).to(device)
    fake_images = generator(edges, attributes)
    print(f"Generator output shape: {fake_images.shape}")

    # Test Discriminator
    print("Testing Discriminator...")
    discriminator = PatchDiscriminator(attr_dim=attr_dim).to(device)
    disc_output = discriminator(edges, attributes, images)
    print(f"Discriminator output shape: {disc_output.shape}")

    # Test Perceptual Loss
    print("Testing Perceptual Loss...")
    perceptual_loss = PerceptualLoss(device=device)
    perc_loss = perceptual_loss(fake_images, images)
    print(f"Perceptual loss: {perc_loss.item()}")

    print("All tests passed!")

    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")