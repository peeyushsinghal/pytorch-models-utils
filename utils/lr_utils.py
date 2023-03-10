# -*- coding: utf-8 -*-
"""lr_utils.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Jz8BNhCK6rY8OQ8F7F-tZM2gP9yv8Xjg
"""

# !pip install torch_lr_finder
from torch_lr_finder import LRFinder
import numpy as np


def find_lr(net, optimizer, criterion, loader):
    """Find learning rate for using One Cyclic LRFinder
    Args:
        net (instace): torch instace of defined model
        optimizer (instance): optimizer to be used
        criterion (instance): criterion to be used for calculating loss
        loader (instance): torch dataloader instance , ideally training
    """

    lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
    lr_finder.range_test(loader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot()
    min_loss = min(lr_finder.history['loss'])
    lr_lowest_loss = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
    print("LR at lowest Loss is {}".format(lr_lowest_loss))

    lr_finder.reset()
    return format(lr_lowest_loss)