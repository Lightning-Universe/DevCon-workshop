import os
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=50_000)
        return train_len

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        if stage == "fit" or stage is None:
            dataset_train = CIFAR10(
                self.data_dir, train=True, transform=train_transforms
            )
            dataset_val = CIFAR10(
                self.data_dir, train=True, transform=self.default_transforms()
            )

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            self.dataset_test = CIFAR10(
                self.data_dir, train=False, transform=self.default_transforms()
            )

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(
            dataset, splits, generator=torch.Generator().manual_seed(self.seed)
        )

        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits

    @abstractmethod
    def default_transforms(self) -> Callable:
        if self.normalize:
            cf10_transforms = transform_lib.Compose(
                [transform_lib.ToTensor(), cifar10_normalization()]
            )
        else:
            cf10_transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return cf10_transforms

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        return self._data_loader(self.dataset_val)

    def test_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )


def cifar10_normalization():
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    return normalize
