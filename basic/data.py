"""
Data loading utilities for CIFAR 10.
"""
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_dataloaders(root, batch_size):
    """
    Return the train, test loader and the number of classes.
    """
    logger.info("Initializing dataloaders")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=root, train=True,
                                download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=root, train=False,
                               download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return trainloader, testloader, 10
