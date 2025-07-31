import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from EffiColorizer.utility import rgb_to_lab


# -----------------------------------------------------------------------------------------------
def train_validation_split(path, train_size, example_size='all', random_state=0):
    paths = glob.glob(path + "/*.jpg")  # grabbing all the jpg files
    rand = np.random.RandomState(random_state)
    if example_size == 'all':
        example_size = len(paths)

    paths_subset = rand.choice(paths, size=example_size, replace=False)
    rand_indexes = rand.permutation(example_size)
    train_paths = paths_subset[rand_indexes[:int(example_size*train_size)]]
    val_paths = paths_subset[rand_indexes[int(example_size*train_size):]]
    return train_paths, val_paths


# -----------------------------------------------------------------------------------------------
class ColorizationDataset(Dataset):
    """
        PyTorch Dataset for image colorization tasks.

        Transformations per split:
            - 'train': Resize to (img_size, img_size) using bicubic interpolation, then apply random horizontal flip with 50% probability.
            - 'val' : Resize only to (img_size, img_size) using bicubic interpolation (no flipping).

        Args:
            paths (list): List of file paths to input RGB images.
            split (str): One of ['train', 'val'] indicating the dataset split.
            size (int): Target size to resize the image (square).

        Returns:
            dict: A dictionary with keys 'L' and 'ab' representing LAB image channels.
    """
    def __init__(self, paths, split, size):
        if split == 'train':
            self.transforms = transforms.Compose([
                # transforms.Resize((size, size),  Image.BICUBIC),  changed!
                # Note: BICUBIC is cubic interpolation applied in two dimensions
                transforms.Resize((size, size), InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'val':
            # self.transforms = transforms.Resize((size, size),  Image.BICUBIC)  changed!
            self.transforms = transforms.Resize((size, size), InterpolationMode.BICUBIC)

        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        L, ab = rgb_to_lab(img)
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


# -----------------------------------------------------------------------------------------------
def make_dataloaders(batch_size=16, shuffle=False, n_workers=0, pin_memory=True, **kwargs):
    """
        Creates a PyTorch DataLoader for the ColorizationDataset.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset.
            n_workers (int): Number of subprocesses for data loading.
            pin_memory (bool): If True, copies data to pinned memory before returning.

        Returns:
            DataLoader: A PyTorch DataLoader instance.
    """
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, pin_memory=pin_memory)
    return dataloader
