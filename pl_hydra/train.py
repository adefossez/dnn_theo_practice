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

from pl_hydra.data import DataModule


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
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
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
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'train_loss' not in metrics:
            # PL makes a first dummy valid only epoch for debugging.
            return
        logger.info(
            'End of epoch %d, train loss %.4f, valid loss %.4f, acc %.1f',
            trainer.current_epoch,
            metrics['train_loss'],
            metrics['valid_loss'],
            100 * metrics['valid_acc'])


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # One important thing to know about Hydra is that it will
    # change automatically the current working directory to a per experiment folder.
    # This can be a bit annoying sometimes, here we need to rewrite the dataset
    # path to be absolute and not relative.
    cfg.data.root = hydra.utils.to_absolute_path(cfg.data.root)
    data = DataModule(**cfg.data)
    module = MainModule(10, cfg)

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


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Make sure any exception is logged here.
        logger.exception("Exception happened during training")
        raise
