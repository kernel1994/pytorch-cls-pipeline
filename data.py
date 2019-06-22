import numpy as np
import torch
from keras.datasets import mnist
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import cfg


class MNISTDataset(Dataset):
    def __init__(self, image_array, label_array, transform=None):
        self.image_array = image_array
        self.label_array = label_array
        self.transform = transform

    def __len__(self):
        return self.image_array.shape[0]

    def __getitem__(self, index):
        image = self.image_array[index]
        label = self.label_array[index]

        if self.transform:
            image = self.transform(image)

        return image, label


def mnist_dataloader():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # convert PyTorch data format (N, C, H, W)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    train_dataloader = DataLoader(MNISTDataset(x_train, y_train, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=cfg.batch_size, shuffle=True)

    test_dataloader = DataLoader(MNISTDataset(x_test, y_test, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=cfg.batch_size, shuffle=False)

    return train_dataloader, test_dataloader
