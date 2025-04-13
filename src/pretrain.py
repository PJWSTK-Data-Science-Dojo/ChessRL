import pytorch_lightning as pl
import torch

from pretrain.utils.data import ChessDataModule
from pretrain.utils.preprocess import FenDataset, LunaPreprocessing
from luna.luna import Luna_Network
from luna.game import ChessGame
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict
from torch import nn, optim

pl.seed_everything(100)


# Create data module
data_module = ChessDataModule(
    data_dir='pretrain/data/out',
    batch_size=1024,
    num_workers=0,  # Don't use workers as it copies dataset and has os.chdir implications
    schema=FenDataset.Schema,
    preprocessing=[  # For e.g. FEN dataset preprocessing has to be done during batch creation
        FenDataset(),  # Converts to standard and flexible dataset representation
        LunaPreprocessing(use_mask=False),  # Converts to Luna sample
    ],
    transforms=[
        # Here space for augmentation etc. Operates on results of preprocessing.
    ],
)


# Create Luna model
net = Luna_Network(
    ChessGame()
)
net.nnet.init_weights()


# Create PyTorch Lightning module wrapper
class ExampleNetLightning(LightningModule):

    def __init__(self, model: Luna_Network, l2_lambda: float, entropy_lambda: float):
        super().__init__()
        self.model = model.nnet
        self.luna = model
        self.l2_lambda = l2_lambda
        self.entropy_lambda = entropy_lambda

    def training_step(self, batch: Dict):
        target_value, label = batch["value"], batch["label"]
        boardAndValid =  batch["state"], batch["mask"]

        policy, value = self.model(boardAndValid)

        # Standard loss
        loss_policy = nn.functional.cross_entropy(policy.clone(), label,
                                                  ignore_index=LunaPreprocessing.BAD_INDEX).mean()
        loss_value = nn.functional.mse_loss(value.flatten(), target_value.flatten()).mean()
        loss_l2 = self.l2_lambda * torch.mean(sum(torch.norm(param, 2) ** 2 for param in self.model.parameters()))

        # Compliment on target distribution being binary (like in PPO)
        loss_entropy = self.entropy_lambda * -torch.sum(policy.clone().detach().softmax(-1)
                                                        * torch.log(policy.softmax(-1) + 1e-8), dim=-1)
        loss_entropy[label == LunaPreprocessing.BAD_INDEX] = 0.0
        loss_entropy = torch.mean(loss_entropy)

        loss = loss_policy + loss_value + loss_l2 + loss_entropy

        self.log('train_loss_policy', loss_policy.clone(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_value', loss_value.clone(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.clone(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_entropy', loss_entropy.clone(), on_step=True, on_epoch=True)
        self.log('train_loss_l2', loss_l2.clone(), on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)


# Run the training
model = ExampleNetLightning(net, l2_lambda=1e-4, entropy_lambda=1e-3)
trainer = Trainer(
    max_epochs=10,
    logger=TensorBoardLogger(
        save_dir="tensorboard",
        name="luna_training",
    ),
    accelerator="gpu",
)
trainer.fit(model, data_module)
torch.save(model, "lightning_model.pt")
