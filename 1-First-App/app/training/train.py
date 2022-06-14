# import pandas as pd
# import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datamodule import CIFAR10DataModule
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.05, batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.hparams.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


def main():
    seed_everything(0)
    model = LitResnet(lr=0.05, batch_size=(256 if torch.cuda.is_available() else 64))
    datamodule = CIFAR10DataModule(
        data_dir="data",
        batch_size=(256 if torch.cuda.is_available() else 64),
        # num_workers=int(os.cpu_count() / 2),
    )

    trainer = Trainer(
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_train_batches=3,
        limit_val_batches=5,
        limit_test_batches=5,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)

    # metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    # del metrics["step"]
    # metrics.set_index("epoch", inplace=True)
    # display(metrics.dropna(axis=1, how="all").head())
    # sn.relplot(data=metrics, kind="line")


if __name__ == "__main__":
    main()
