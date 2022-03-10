# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from pl.data import DataModule


logger = logging.getLogger(__name__)


class MainModule(pl.LightningModule):

    def __init__(self, num_classes, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = getattr(models, cfg.model)(num_classes=num_classes)
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = F.cross_entropy(scores, y)
        self.log('train_loss', loss, per_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = F.cross_entropy(scores, y)
        acc = (y == scores.argmax(-1)).float().mean()
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), **self.cfg.optim)
        return optimizer


class MyLogger(Callback):
    def on__epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print(metrics.keys())
        logger.info(
            'End of epoch %d, valid loss %.4f, acc %.2%',
            trainer.current_epoch,
            metrics['valid_loss'],
            metrics['valid_acc'])


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # One important thing to know about Hydra is that it will
    # change automatically the current working directory to a per experiment folder.
    # This can be a bit annoying sometimes, here we need to rewrite the dataset
    # path to be absolute and not relative.
    cfg.data.root = hydra.utils.to_absolute_path(cfg.data.root)
    data = DataModule(**cfg.data)
    module = MainModule(10)

    root = Path('.')
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_last=True)
    resume_from_checkpoint = None
    if not cfg.restart:
        last_checkpoint_path = root / 'last.ckpt'
        if last_checkpoint_path.exists():
            resume_from_checkpoint = last_checkpoint_path
    trainer = pl.Trainer(
        **cfg.trainer,
        default_root_dir=root,
        resume_from_checkpoint=resume_from_checkpoint,
        callbacks=[
            checkpoint_callback,
            MyLogger(),
        ],

    )
    trainer.fit(module, data)