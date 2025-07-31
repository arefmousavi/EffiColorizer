import torch
import torch.nn as nn
from EffiColorizer.utility import conv_block, conv_transpose_block
import segmentation_models_pytorch as smp


class Generator_V1(nn.Module):
    """
        Creates a U-Net generator model for image colorization task.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale images).
            out_channels (int): Output channels of the final layer (usually 2 for ab channels in Lab color space).
            filter_sizes (list of int): A list of 8 integers defining the number of filters at each encoder and decoder layer.
                                        The length must be exactly 8, corresponding to the depth of the U-Net.

        Returns:
            The output image tensor with shape (N, out_channels, H, W), where H and W match the input size.
    """
    def __init__(self, in_channels, out_channels, filter_sizes):
        super().__init__()

        if len(filter_sizes) != 8:
            raise ValueError("length of filter_sizes must be 8")

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder blocks
        self.encoder.append(conv_block(in_f=in_channels, out_f=filter_sizes[0],
                                       kernel_size=5, stride=1, padding=2, batch_norm=True, activation='LeakyReLU'))

        enc_args = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'batch_norm': True, 'activation': 'LeakyReLU'}

        for i in range(1, 8):
            self.encoder.append(conv_block(in_f=filter_sizes[i - 1], out_f=filter_sizes[i], **enc_args))

        # Decoder blocks
        dec_args = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'batch_norm': True, 'activation': 'ReLU'}

        self.decoder.append(conv_transpose_block(in_f=filter_sizes[7], out_f=filter_sizes[6], **dec_args))

        for i in range(6, 0, -1):
            self.decoder.append(conv_transpose_block(in_f=2 * filter_sizes[i], out_f=filter_sizes[i - 1], **dec_args))

        self.decoder.append(conv_transpose_block(in_f=2 * filter_sizes[0], out_f=out_channels,
                                                 kernel_size=5, stride=1, padding=2, batch_norm=False, activation='Tanh'))

    def forward(self, x):
        enc_outs = []

        for layer in self.encoder:
            x = layer(x)
            enc_outs.append(x)

        x = self.decoder[0](enc_outs[-1])

        for i in range(1, 8):
            skip = enc_outs[-(i + 1)]
            x = self.decoder[i](torch.cat([skip, x], dim=1))

        return x


def generator_V2(img_size):
    """
        Creates a U-Net generator model for image colorization task.

        Args:
            img_size (int): The input image size. Must be 320 (raises exception otherwise).

        Returns:
            A U-Net generator model with:
                - 1 input channel.
                - 2 output channels with 'tanh' activation.
                - Pretrained EfficientNet-B3 encoder.
    """
    if img_size != 320:
        raise Exception("img size must be 320")
    net_G = smp.Unet(encoder_name="efficientnet-b3", encoder_weights="imagenet", in_channels=1, classes=2, activation='tanh',
                     decoder_channels=[256, 128, 64, 32, 16], encoder_depth=5)
    return net_G
