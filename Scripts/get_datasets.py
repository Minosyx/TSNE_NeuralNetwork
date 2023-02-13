import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda, Normalize
from torch import flatten
import os
from typing import Tuple, List


def get_mnist() -> Tuple[Dataset, Dataset]:
    mnist_dataset_train = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )

    mnist_dataset_test = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )
    return mnist_dataset_train, mnist_dataset_test


def get_fashion_mnist() -> Tuple[Dataset, Dataset]:
    fashion_mnist_dataset_train = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )

    fashion_mnist_dataset_test = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )

    return fashion_mnist_dataset_train, fashion_mnist_dataset_test


def _get_available_datasets() -> List[str]:
    methods = [key[4:] for key in globals().keys() if key.startswith("get")]
    methods.remove("dataset")
    return methods


def prepare_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    if not (
        os.path.exists(dataset_name + "_train.data")
        and os.path.exists(dataset_name + "_test.data")
    ):
        train, test = globals()["get_" + dataset_name]()
        torch.save(train, dataset_name + "_train.data")
        torch.save(test, dataset_name + "_test.data")
    else:
        train = torch.load(dataset_name + "_train.data")
        test = torch.load(dataset_name + "_test.data")
    return train, test


def get_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    name = dataset_name.lower()
    if name in _get_available_datasets():
        return prepare_dataset(name)
    return None, None
