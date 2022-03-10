"""This is an example of a complete training pipeline with logging and checkpointing,
using only vanilla PyTorch. While frameworks like PyTorch-Lightning (PL) can take care
of most of this for you, it is a good idea to understand what goes on under the hood.
In the future, you might have to work with codebases that do not use PL, or you might
want to have more control over what is going on.

You can run this script from the root directory as
```
python -m basic.train
```
One advantage of launching like that rather than
```
python basic/train.py
```
is that the `basic` folder will automatically be inside the Python path.
"""
# Typical order for imports is builtin -> other packages -> current package.
import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

import colorlog
import torch
import torchvision.models as models
from torch.nn import functional as F

# Avoid wildcard imports (from basic.data import *) as they make it very
# hard to know which function is coming from which file for collaborators.
from basic.data import get_dataloaders
from basic.utils import averager, write_and_rename


logger = logging.getLogger('train')


def setup_logging(xp_folder):
    """Setup logging, with one log to the stdout, and one to a train.log file.
    """
    # See https://docs.python.org/3/howto/logging-cookbook.html#logging-cookbook for reference.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fh = logging.FileHandler(xp_folder / 'train.log')
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    formatter = colorlog.ColoredFormatter(
        '[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]'
        '[%(log_color)s%(levelname)s%(reset)s] %(message)s',
        datefmt='%m-%d %H:%M:%S',  # removing milliseconds from log,
                                   # they take space and are rarely useful
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
    )
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    root_logger.addHandler(fh)
    root_logger.addHandler(sh)


def get_parser():
    """Return the parser for commandline args.
    """
    # For now we use argparse to handle options passed to the training script.
    # For more complex project, advanced solutions like Hydra are usually
    # prefered (https://github.com/facebookresearch/hydra)
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--data', default='data/', help='Root for data storage.')

    parser.add_argument('--storage', default=Path('./outputs'), type=Path,
                        help='Where experiments are stored.')
    parser.add_argument('-R', '--restart', action='store_true',
                        help='Wipes out previous checkpoints or logs.')

    return parser


# Those args will not be used for generating the experiment name.
IGNORED_FOR_NAME = [
    'restart',
    'storage',
    'data',
]


def get_name(args, parser):
    """Generate an experiment name from the args and parser."""
    # We need each experiment to have a name in order to easily find the logs,
    # this will also allow to automatically resume any previous checkpoint.
    parts = []
    name_args = dict(args.__dict__)
    for name, value in name_args.items():
        if name in IGNORED_FOR_NAME:
            continue
        if value != parser.get_default(name):
            if isinstance(value, Path):
                parts.append(f"{name}={value.name}")
            else:
                parts.append(f"{name}={value}")
    if parts:
        name = " ".join(parts)
    else:
        name = "default"
    return name


def do_epoch(epoch, model, loader, optimizer=None):
    """Run a single epoch, either in training or evaluation mode, if `optimizer` is None."""

    device = next(model.parameters()).device

    average = averager()

    for input_, label in loader:
        input_ = input_.to(device)
        label = label.to(device)

        prediction = model(input_)

        loss = F.cross_entropy(prediction, label)
        predicted_label = prediction.argmax(dim=1)
        accuracy = (label == predicted_label).float().mean()

        metrics = {
            'loss': loss,
            'accuracy': accuracy,
        }
        metrics = average(metrics)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    label = 'test' if optimizer is None else 'train'
    logger.info(f'Epoch {epoch:03d} {label: <5} summary '
                f'loss: {metrics["loss"]:.3f}, '
                f'acc.: {metrics["accuracy"]:6.2%}')
    return metrics


def main():
    # Always wrap the training code in a main function, as otherwise
    # it will get executed when spawning subprocesses (Dataloader workers, Distributed processes)

    parser = get_parser()
    args = parser.parse_args()

    # Make sure XP folder exists
    name = get_name(args, parser)
    xp_folder = args.storage / name
    if args.restart and xp_folder.exists():
        shutil.rmtree(xp_folder)
    xp_folder.mkdir(exist_ok=True, parents=True)
    checkpoint_path = xp_folder / 'checkpoint.th'
    history_file = xp_folder / 'history.json'  # convenient way to get metrics for later plotting

    setup_logging(xp_folder)
    logger.info("This is experiment %s", name)
    logger.info("Checkout %s for logs and checkpoints", xp_folder)

    trainloader, testloader, num_classes = get_dataloaders(args.data, args.batch_size)

    model = getattr(models, args.model)(num_classes=num_classes)

    # Move model to GPU if cuda is available. If you have multiple GPU and want
    # to select one, you can run with `CUDA_VISIBLE_DEVICES=1 ./train.py` (GPU indexes starts at 0)
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []  # keep track of all metrics
    if checkpoint_path.exists():
        # map_location=cpu, avoid issues when loading checkpoints from machines with different
        # numbers of GPUs.
        logger.info("Loading checkpoint %s", checkpoint_path)
        pkg = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(pkg['model_state'])
        optimizer.load_state_dict(pkg['optimizer_state'])
        history = pkg['history']

    for epoch in range(len(history), args.epochs):
        model.train()
        train_metrics = do_epoch(epoch, model, trainloader, optimizer)
        model.eval()
        test_metrics = do_epoch(epoch, model, testloader)

        history.append({
            'train': train_metrics,
            'test': test_metrics,
        })

        pkg = {
            'history': history,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            # Storing args is a great idea for the future. You will most likely store
            # somewhere about your best checkpoints for a research project, then forget
            # all about it for a year, find it back and wonder how it was train :)
            'args': args,
        }
        with write_and_rename(checkpoint_path) as tmp_file:
            # Saving directly on the checkpoint path is risky: if the job gets interrupted
            # while writing the checkpoint, you will lose everything: the old checkpoints
            # is overwritten but the new one is not fully written.
            # This can especially happen for larger models.
            torch.save(pkg, tmp_file)

        # This one is not as critical, as the full history is still stored
        # in the checkpoint file.
        json.dump(history, open(history_file, 'w'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Make sure any exception is logged here.
        logger.exception("Exception happened during training")
        raise
