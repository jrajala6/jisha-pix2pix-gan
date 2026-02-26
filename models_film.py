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


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Projects an attribute vector into per-channel scale (γ) and shift (β),
    then applies  features * (1 + γ) + β  element-wise.
    Zero-initialized so FiLM starts as identity (no disruption early in training).
    """

    def __init__(self, attr_dim: int, channels: int, hidden: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(attr_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),  # γ and β
        )
        # Zero-init: FiLM starts as identity transform
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, features: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        params = self.proj(attributes)                     # [B, 2C]
        gamma, beta = params.chunk(2, dim=1)               # each [B, C]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return features * (1 + gamma) + beta


class AttributeInjector(nn.Module):
    """Inject attribute vector into feature maps using adaptive normalization (bottleneck)"""

    def __init__(self, feature_dim: int, attr_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.attr_dim = attr_dim

        self.attr_proj = nn.Sequential(
            nn.Linear(attr_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )

    def forward(self, features: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape

        attr_params = self.attr_proj(attributes)
        scale, bias = torch.chunk(attr_params, 2, dim=1)

        scale = scale.view(B, C, 1, 1)
        bias = bias.view(B, C, 1, 1)

        if H > 1 and W > 1:
            features_norm = F.instance_norm(features)
        else:
            features_norm = features

        return features_norm * (1 + scale) + bias


class Generator(nn.Module):
    """
    U-Net Generator with FiLM conditioning at every decoder layer
    + attribute injection at bottleneck.

    Attributes influence the output at ALL spatial resolutions,
    from 2×2 up to 128×128.
    """

    def __init__(self, attr_dim: int, ngf: int = 64):
        super().__init__()

        self.attr_dim = attr_dim

        # Encoder (downsampling)
        self.e1 = nn.Conv2d(3, ngf, 4, 2, 1)
        self.e2 = UNetBlock(ngf, ngf * 2, down=True, use_norm=True)
        self.e3 = UNetBlock(ngf * 2, ngf * 4, down=True, use_norm=True)
        self.e4 = UNetBlock(ngf * 4, ngf * 8, down=True, use_norm=True)
        self.e5 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=True)
        self.e6 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=True)
        self.e7 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=True)

        # Bottleneck
        self.bottleneck = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=False)

        # Attribute injection at bottleneck (adaptive normalization)
        self.attr_injector = AttributeInjector(ngf * 8, attr_dim)

        # Decoder (upsampling)
        self.d1 = UNetBlock(ngf * 8, ngf * 8, down=False, use_dropout=True)
        self.d2 = UNetBlock(ngf * 16, ngf * 8, down=False, use_dropout=True)
        self.d3 = UNetBlock(ngf * 16, ngf * 8, down=False, use_dropout=True)
        self.d4 = UNetBlock(ngf * 16, ngf * 8, down=False, use_dropout=False)
        self.d5 = UNetBlock(ngf * 16, ngf * 4, down=False, use_dropout=False)
        self.d6 = UNetBlock(ngf * 8, ngf * 2, down=False, use_dropout=False)
        self.d7 = UNetBlock(ngf * 4, ngf, down=False, use_dropout=False)

        # FiLM layers — one per decoder block, applied to OUTPUT features
        self.film1 = FiLMLayer(attr_dim, ngf * 8)   # d1 output: 512ch @ 2×2
        self.film2 = FiLMLayer(attr_dim, ngf * 8)   # d2 output: 512ch @ 4×4
        self.film3 = FiLMLayer(attr_dim, ngf * 8)   # d3 output: 512ch @ 8×8
        self.film4 = FiLMLayer(attr_dim, ngf * 8)   # d4 output: 512ch @ 16×16
        self.film5 = FiLMLayer(attr_dim, ngf * 4)   # d5 output: 256ch @ 32×32
        self.film6 = FiLMLayer(attr_dim, ngf * 2)   # d6 output: 128ch @ 64×64
        self.film7 = FiLMLayer(attr_dim, ngf)        # d7 output: 64ch  @ 128×128

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, edges: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = F.leaky_relu(self.e1(edges), 0.2)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        # Bottleneck + attribute injection
        bottleneck = self.bottleneck(e7)
        bottleneck = self.attr_injector(bottleneck, attributes)

        # Decoder with skip connections + FiLM at every layer
        d1 = self.film1(self.d1(bottleneck), attributes)
        d2 = self.film2(self.d2(torch.cat([d1, e7], dim=1)), attributes)
        d3 = self.film3(self.d3(torch.cat([d2, e6], dim=1)), attributes)
        d4 = self.film4(self.d4(torch.cat([d3, e5], dim=1)), attributes)
        d5 = self.film5(self.d5(torch.cat([d4, e4], dim=1)), attributes)
        d6 = self.film6(self.d6(torch.cat([d5, e3], dim=1)), attributes)
        d7 = self.film7(self.d7(torch.cat([d6, e2], dim=1)), attributes)

        # Final output
        output = self.final(torch.cat([d7, e1], dim=1))

        return output


class PatchDiscriminator(nn.Module):
    """PatchGAN Discriminator with spectral normalization (same as baseline)."""

    def __init__(self, attr_dim: int, ndf: int = 64):
        super().__init__()

        self.attr_dim = attr_dim

        # Lightweight attribute conditioning: project to scalar, broadcast spatially
        self.attr_proj = nn.Sequential(
            nn.Linear(attr_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.conv1 = spectral_norm(nn.Conv2d(7, ndf, 4, 2, 1))

        self.conv2 = spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1))
        self.norm2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1))
        self.norm3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1))
        self.norm4 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1))

        self.apply(weights_init)

    def forward(self, edges, attributes, images):
        B, _, H, W = images.shape

        attr_scalar = self.attr_proj(attributes)          # [B, 1]
        attr_map = attr_scalar.view(B, 1, 1, 1).expand(B, 1, H, W)  # broadcast

        x = torch.cat([edges, images, attr_map], dim=1)

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

        output = self.conv5(x)

        return output


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features."""

    def __init__(self, feature_layers=None, device='cuda'):
        super().__init__()

        if feature_layers is None:
            feature_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

        self.feature_layers = feature_layers
        self.device = device

        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg.to(device)
        self.vgg.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.layer_name_mapping = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }

    def forward(self, pred, target):
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0

        pred = self.normalize(pred)
        target = self.normalize(target)

        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = 0.0
        for layer in self.feature_layers:
            if layer in pred_features and layer in target_features:
                loss += F.mse_loss(pred_features[layer], target_features[layer])

        return loss

    def _extract_features(self, x):
        features = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.feature_layers:
                    features[layer_name] = x

        return features


def hinge_loss_generator(fake_output):
    return -torch.mean(fake_output)


def hinge_loss_discriminator(real_output, fake_output):
    real_loss = torch.mean(F.relu(1.0 - real_output))
    fake_loss = torch.mean(F.relu(1.0 + fake_output))
    return real_loss + fake_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing FiLM models on {device}")

    batch_size = 4
    attr_dim = 102
    image_size = 256

    edges = torch.randn(batch_size, 3, image_size, image_size).to(device)
    attributes = torch.randn(batch_size, attr_dim).to(device)
    images = torch.randn(batch_size, 3, image_size, image_size).to(device)

    print("Testing FiLM Generator...")
    generator = Generator(attr_dim=attr_dim).to(device)
    fake_images = generator(edges, attributes)
    print(f"Generator output shape: {fake_images.shape}")

    print("Testing Discriminator...")
    discriminator = PatchDiscriminator(attr_dim=attr_dim).to(device)
    disc_output = discriminator(edges, attributes, images)
    print(f"Discriminator output shape: {disc_output.shape}")

    print("Testing Perceptual Loss...")
    perceptual_loss = PerceptualLoss(device=device)
    perc_loss = perceptual_loss(fake_images, images)
    print(f"Perceptual loss: {perc_loss.item()}")

    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"\nGenerator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")

    # Count FiLM-only parameters
    film_params = sum(
        p.numel() for name, p in generator.named_parameters()
        if 'film' in name and p.requires_grad
    )
    print(f"FiLM layer parameters: {film_params:,} ({100*film_params/g_params:.1f}% of G)")

    print("\nAll tests passed!")
