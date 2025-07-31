from torch import nn
from EffiColorizer.utility import conv_block


class Discriminator(nn.Module):
    """
        A flexible convolutional discriminator for GANs.

        Architecture:
            - Intermediate layers: Convolution + Spectral Norm + LeakyReLU
            - Final layer: Convolution + Spectral Norm (no activation)

        Args:
            in_channels (int): Number of input channels (e.g., 3 for Lab images).
            out_channels (int): Output channels of the final layer (usually 1).
            filter_sizes (list of int): List of intermediate layer filter sizes.

        Returns:
            - The final output feature maps.
    """
    def __init__(self, in_channels, out_channels, filter_sizes):
        super().__init__()
        sizes = [in_channels, *filter_sizes, out_channels]
        n_layers = len(filter_sizes) + 1  # number of layers (num of filters + final layer)
        layers = []
        for in_f, out_f in zip(sizes, sizes[1:]):
            layers.append({'in_f': in_f, 'out_f': out_f})

        blocks = []

        blocks += [conv_block(in_f=layer['in_f'], out_f=layer['out_f'],
                   kernel_size=4, stride=2, padding=1, batch_norm=False, activation='LeakyReLU', spectral_norm=True)
                   for layer in layers[:n_layers-1]]

        blocks += [conv_block(in_f=layers[n_layers-1]['in_f'], out_f=layers[n_layers-1]['out_f'],
                   kernel_size=5, stride=1, padding=2, batch_norm=False, activation='None', spectral_norm=True)]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
