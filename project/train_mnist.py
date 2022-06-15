import torch
import torch.nn as nn
from datamodule import MNISTDataModule
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
from torchmetrics import Accuracy


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ImageClassifier(LightningModule):
    def __init__(self, lr=1.0, gamma=0.7, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.model = Net()
        self.test_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], [
            torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=self.hparams.gamma
            )
        ]


def main():
    seed_everything(0)
    model = ImageClassifier()
    datamodule = MNISTDataModule(data_dir="data")
    trainer = Trainer(max_epochs=2)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
