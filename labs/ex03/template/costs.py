# -*- coding: utf-8 -*-
import numpy as np

"""Function used to compute the loss."""


def compute_loss_mse(y, tx, w):
    """MAE"""
    e = y - tx.dot(w)
    return 0.5 * (np.linalg.norm(e) ** 2) / len(y)


def compute_loss_mae(y, tx, w):
    """MSE"""
    e = y - tx.dot(w)
    mae = 0.5 * (np.linalg.norm(e, 1)) / len(y)
    return mae
