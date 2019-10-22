# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse


def least_squares(y, tx):
    """calculate the least squares solution."""
    w_star = np.linalg.inv(tx.T @ tx) @ (tx.T) @ y
    loss = compute_mse(y, tx, w_star)
    return loss, w_star
