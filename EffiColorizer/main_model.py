import torch
from torch import nn, optim
from EffiColorizer.utility import set_requires_grad, batch_lab_to_rgb


class MainModel(nn.Module):
    """
        This class manages both training and inference processes of the entire model.

        Architecture:
            - Generator predicts color channels (ab) from grayscale input (L).
            - Discriminator evaluates realism of colorized images.
            - Supports multi-loss training including L1 and adversarial GAN losses.
            - Allows flexible control over training generator and discriminator independently.

        Args:
            net_G (nn.Module): Generator network.
            net_D (nn.Module): Discriminator network.
            device (torch.device): Device for computation.

        Returns:
            - During training: dictionary of computed losses for generator and discriminator.
            - During inference: colorized RGB images.
    """
    def __init__(self, net_G, net_D, device):
        super().__init__()
        self.device = device
        self.net_G = net_G.to(self.device)
        self.net_D = net_D.to(self.device)

        self.loss_G_GAN = None
        self.loss_G_L1 = None
        self.loss_G = None
        self.loss_D_fakes = None
        self.loss_D_reals = None
        self.loss_D = None
        self.GAN_criterion = None
        self.L1_criterion = None
        self.opt_G = None
        self.opt_D = None
        self.lambda_L1 = 0
        self.train_G = False
        self.train_D = False

    def _load_data(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def _calculate_grad_G(self):
        # freeze discriminator (don't save gradient for discriminator, it's not needed)
        set_requires_grad(self.net_D, False)
        # generator loss
        fake_colors = self.net_G(self.L)
        fake_images = torch.cat([self.L, fake_colors], dim=1)
        fake_predictions = self.net_D(fake_images)
        self.loss_G_GAN = self.GAN_criterion(predictions=fake_predictions, label='real')
        self.loss_G_L1 = self.L1_criterion(input=fake_colors, target=self.ab)
        self.loss_G = self.loss_G_GAN + (self.loss_G_L1 * self.lambda_L1)
        self.loss_G.backward()
        # unfreeze discriminator
        set_requires_grad(self.net_D, True)

    def _calculate_grad_D(self):
        # fake images
        fake_colors = self.net_G(self.L)
        fake_images = torch.cat([self.L, fake_colors], dim=1)
        fake_predictions = self.net_D(fake_images.detach())
        self.loss_D_fake = self.GAN_criterion(predictions=fake_predictions, label='fake')
        # real images
        real_images = torch.cat([self.L, self.ab], dim=1)
        real_predictions = self.net_D(real_images)
        self.loss_D_real = self.GAN_criterion(real_predictions, label='real')
        # discriminator loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) / 2
        self.loss_D.backward()

    def _get_losses(self):
        return {'loss_G_GAN':  self.loss_G_GAN.item(),
                'loss_G_L1':   self.loss_G_L1.item(),
                'loss_G':      self.loss_G.item(),
                'loss_D_fake': self.loss_D_fake.item(),
                'loss_D_real': self.loss_D_real.item(),
                'loss_D':      self.loss_D.item()
                }

    def set_hyperparameters(self, GAN_loss=None, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.,
                                  train_G=True, train_D=True):
        self.GAN_criterion = GAN_loss
        self.L1_criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.lambda_L1 = lambda_L1
        self.train_G = train_G
        self.train_D = train_D

    def forward(self, L):
        self.net_G.eval()
        L = L.to(self.device)
        # Forward pass in inference mode (no gradient computation)
        with torch.no_grad():
            fake_colors = self.net_G(L)

        fake_images = batch_lab_to_rgb(L, fake_colors)
        return fake_images

    def train_GAN(self, data):
        # load imgs
        self._load_data(data)
        # update discriminator
        if self.train_D:
            self.net_D.train()  # Switch to training mode
            self.opt_D.zero_grad()
            self._calculate_grad_D()
            self.opt_D.step()
        # update generator
        if self.train_G:
            self.net_G.train()
            self.opt_G.zero_grad()
            self._calculate_grad_G()
            self.opt_G.step()
        # return a dict of all losses
        return self._get_losses()

    def pretrain_G(self, data):
        # load imgs
        self._load_data(data)
        # pretrain generator with L1-loss
        self.net_G.train()
        fake_colors = self.net_G(self.L)
        self.loss_G_L1 = self.L1_criterion(input=fake_colors, target=self.ab)
        self.opt_G.zero_grad()
        self.loss_G_L1.backward()
        self.opt_G.step()
        # return L1-loss
        return self.loss_G_L1.item()
