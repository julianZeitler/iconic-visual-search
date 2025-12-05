import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from cvtools.models.pytorch import PyTorchModel, L2Norm

class ArcLayer(nn.Module):

    def __init__(self, embedding_size, n_classes):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(embedding_size, n_classes))

        nn.init.kaiming_normal_(self.weights)


    def forward(self, x):
        weights = F.normalize(self.weights, p=2, dim=0)

        return torch.mm(x, weights)

class VAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=12):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_fc = nn.Linear(self.input_dim, 128)
        self.mu = nn.Linear(128, self.latent_dim)
        self.logvar = nn.Linear(128, self.latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(self.latent_dim, 128)
        self.dec_fc2 = nn.Linear(128, self.input_dim)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_fc(x))
        mean = self.mu(h)
        log_var = self.logvar(h)

        z = self.sample_z(mean, log_var)
        return z, mean, log_var
    
    def sample_z(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        assert mean.shape == log_var.shape
        std = torch.exp(0.5 * log_var)
        normal_dist = torch.distributions.Normal(mean, std)
        return normal_dist.rsample()
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        h = self.dec_fc2(h)
        return h
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mean, log_var = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z, mean, log_var


def vae_loss_fn(x: torch.Tensor, x_reconstructed: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor, beta: float = 1e-3, target_std: float = 0.1) -> torch.Tensor:
    """
    Loss function for VAE. Combination of cross-entropy and KL divergence for latent space regularization.

    Args:
        x (torch.Tensor): Input into VAE. Tensor of shape (batch_size, input_dim).
        x_reconstructed (torch.Tensor): Output of VAE. Tensor of shape (batch_size, input_dim).
        mean (torch.Tensor): Latent space mean. Tensor of shape(batch_size, latent_dim).
        log_var (torch.Tensor): Latent space log-variance. Tensor of shape(batch_size, latent_dim).

    Returns:
        loss (torch.Tensor). Combined loss. Tensor of shape ( )

    """
    # Reconstruction loss
    bce_loss = nn.BCEWithLogitsLoss()
    recon_loss = bce_loss(x_reconstructed, x)

    # Modified KL-divergence that allows the mean to be arbitrary
    target_var = target_std ** 2
    kl_loss = 0.5 * torch.sum(log_var.exp()/target_var + mean.pow(2)/target_var - 1 - log_var + torch.log(torch.tensor(target_var)))
    return recon_loss + beta*kl_loss


class BaseCNNModel(PyTorchModel):
    
    def __init__(self, embedding_dim, n_classes, classifier="linear"):
        super().__init__()

        self.outputs = {}
        self.features: nn.Sequential

        if classifier == "linear":
            self.classifier = nn.Linear(embedding_dim, n_classes)
        elif classifier == "arcface":
            self.classifier = ArcLayer(embedding_dim, n_classes)


    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


    def _init_weights(self, submodule=None):

        if submodule is None:
            submodule = self

        for name, module in submodule.named_modules():

            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def _register_hooks(self, layers):
        
        def get_hook(name):
            def hook(module, input, output):
                self.outputs[name] = output
            return hook
        
        for layer in layers:
            getattr(self, layer).register_forward_hook(get_hook(layer))


    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False


    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

class ConvGist(BaseCNNModel):

    def __init__(self, in_channels, n_classes, classifier="linear", output_layers=[]):
        super().__init__(embedding_dim=512, n_classes=n_classes, classifier=classifier)

        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        if classifier == "arcface":
            self.features.append(L2Norm())

        self._init_weights()
        self._register_hooks(output_layers)


class VGG(BaseCNNModel):

    def __init__(self, n_classes, classifier="linear", output_layers=['features']):
        # VGG16 outputs 512 features after global average pooling
        super().__init__(embedding_dim=512, n_classes=n_classes, classifier=classifier)

        # Load pretrained VGG16
        vgg16 = models.vgg16(weights="IMAGENET1K_V1")

        # Extract features (all convolutional layers)
        # We'll modify the last part to add global average pooling
        self.features = nn.Sequential(
            *list(vgg16.features)[:30],  # All conv layers from pretrained VGG until layer 30 (skip max pool layer at the end)
            vgg16.avgpool, # Average pooling with size (7, 7)
            # nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten() # 7x7x512=25088 features
        )

        if classifier == "arcface":
            self.features.append(L2Norm())

        self._register_hooks(output_layers)

class AlexNet(BaseCNNModel):
    def __init__(self, n_classes, classifier="linear", output_layers=['features']):
        super().__init__(embedding_dim=512, n_classes=n_classes, classifier=classifier)

        alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

        self.features = nn.Sequential(
            *list(alexnet.features)[:-1],
            alexnet.avgpool, # (6,6)
            nn.Flatten() # 6x6x256 = 9216 features
        )

        if classifier == "arcface":
            self.features.append(L2Norm())

        self._register_hooks(output_layers)