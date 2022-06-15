import os

from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def train():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        limit_train_batches=2,
        limit_val_batches=2,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_model_summary=False,
        accelerator="cpu",
        num_nodes=2,
        devices=1,
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)


class Work(LightningWork):
    def __init__(self, cloud_compute: CloudCompute = CloudCompute(), **kwargs):
        super().__init__(parallel=True, **kwargs, cloud_compute=cloud_compute)

    def run(self, main_address="localhost", main_port=1111, world_size=1, rank=0, dry_run=False):
        if dry_run:
            return

        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        os.environ["MASTER_ADDR"] = main_address
        os.environ["MASTER_PORT"] = str(main_port)
        os.environ["NODE_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        train()


class MultiNodeDemo(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work0 = Work()
        self.work1 = Work()

    def run(self):
        self.work0.run(dry_run=True)

        if self.work0.internal_ip:
            self.work0.run(main_address=self.work0.internal_ip, main_port=self.work0.port, world_size=2, rank=0)
            self.work1.run(main_address=self.work0.internal_ip, main_port=self.work0.port, world_size=2, rank=1)


app = LightningApp(MultiNodeDemo())
