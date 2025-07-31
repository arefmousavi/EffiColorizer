import math
from torch import Tensor
from PIL.Image import Image as Img
from typing import Tuple, List
import matplotlib.pyplot as plt
from PIL import Image
from prettytable import PrettyTable
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
from torchview import draw_graph
from torch import nn


# -----------------------------------------------------------------------------------------------
def rgb_to_lab(img: Img) -> Tuple[Tensor, Tensor]:
    # Note: L -> (1, img_size, img_size), ab -> (2, img_size, img_size)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    # Normalize
    L = img_lab[[0], ...] / 50. - 1.  # [0, 100] -> [0, 2] -> [-1, 1]
    ab = img_lab[[1, 2], ...] / 128.  # [-127, 128] -> [-0.992, 1]
    return L, ab


# -----------------------------------------------------------------------------------------------
def lab_to_rgb(L: Tensor, ab: Tensor) -> Img:
    # Note: L -> (1, img_size, img_size), ab -> (2, img_size, img_size)
    L = (L + 1.) * 50.  # [-1, 1] -> [0, 100]
    ab = ab * 127.      # [-1, 1] -> [-127, 127]
    img = torch.cat((L, ab), dim=0).permute(1, 2, 0).cpu().numpy()
    img = lab2rgb(img)
    img = Image.fromarray((img*255).astype('uint8'))
    return img


# -----------------------------------------------------------------------------------------------
def batch_lab_to_rgb(L: Tensor, ab: Tensor) -> List[Img]:
    imgs = []
    for i in range(len(L)):
        imgs.append(lab_to_rgb(L[i], ab[i]))
    return imgs


# -----------------------------------------------------------------------------------------------
def batch_rgb_to_lab(imgs: List[Img]) -> Tuple[Tensor, Tensor]:
    L_list = []
    ab_list = []
    for img in imgs:
        L, ab = rgb_to_lab(img)
        L_list.append(L)
        ab_list.append(ab)
    L = torch.Tensor(L_list)
    ab = torch.Tensor(ab_list)
    return L, ab


# -----------------------------------------------------------------------------------------------
def open_imgs(paths: list) -> List[Img]:
    imgs = []
    for path in paths:
        imgs.append(Image.open(path).convert('RGB'))
    return imgs


# -----------------------------------------------------------------------------------------------
def plot_imgs(imgs: list, ncols=4, figsize=(12, 12), title=""):
    nrows = math.ceil(len(imgs)/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, img in zip(axes.flatten(), imgs):
        ax.imshow(img)


# -----------------------------------------------------------------------------------------------
def show_parameters(model, show_elements=3):
    table = PrettyTable()
    table.field_names = ['layers', 'size', f'first {show_elements} elements', 'grad']
    table.align = 'l'
    for name, param in model.named_parameters():
        values = param.data.view(-1)[:show_elements].tolist()
        table.add_row([name, list(param.size()), np.around(values, decimals=4), param.requires_grad])
    print(table)


# -----------------------------------------------------------------------------------------------
def set_requires_grad(model, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad


# -----------------------------------------------------------------------------------------------
def draw_model(model, input_size, depth=2, directory=None, filename=None):
    model_graph = draw_graph(model, input_size=input_size, depth=depth, expand_nested=True)
    if filename and directory:
        model_graph.visual_graph.render(filename=filename, directory=directory, format='pdf', cleanup=True)
    return model_graph.visual_graph


# -----------------------------------------------------------------------------------------------
def conv_block(in_f, out_f, kernel_size=4, stride=2, padding=1, batch_norm=True, activation='None', spectral_norm=False):
    conv_bias = False if batch_norm else True

    if spectral_norm:
        layers = [nn.utils.spectral_norm(nn.Conv2d(in_f, out_f, kernel_size, stride, padding, bias=conv_bias))]
    else:
        layers = [nn.Conv2d(in_f, out_f, kernel_size, stride, padding, bias=conv_bias)]

    if batch_norm:
        layers += [nn.BatchNorm2d(out_f)]

    if activation == 'LeakyReLU':
        layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
    elif activation == 'ReLU':
        layers += [nn.ReLU(inplace=True)]
    elif activation == 'Tanh':
        layers += [nn.Tanh()]
    elif activation == 'Sigmoid':
        layers += [nn.Sigmoid()]
    elif activation == 'None':
        pass
    else:
        raise Exception(f'{activation} is not valid activation name')

    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------------------------
def conv_transpose_block(in_f, out_f, kernel_size=4, stride=2, padding=1, batch_norm=True, activation='None'):
    if batch_norm:
        layers = [nn.ConvTranspose2d(in_f, out_f, kernel_size, stride, padding, bias=False)]
        layers += [nn.BatchNorm2d(out_f)]
    else:
        layers = [nn.ConvTranspose2d(in_f, out_f, kernel_size, stride, padding, bias=True)]

    if activation == 'LeakyReLU':
        layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
    elif activation == 'ReLU':
        layers += [nn.ReLU(inplace=True)]
    elif activation == 'Tanh':
        layers += [nn.Tanh()]
    elif activation == 'Sigmoid':
        layers += [nn.Sigmoid()]
    elif activation == 'None':
        pass
    else:
        raise Exception(f'{activation} is not valid activation name')

    return nn.Sequential(*layers)
