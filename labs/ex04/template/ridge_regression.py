# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_bis = lambda_ * (2 * len(y))
    fst_term = np.linalg.inv(tx.T @ tx + lambda_bis * np.identity(tx.shape[1]))
    snd_term = tx.T @ y
    w_star = fst_term @ snd_term

    return w_star
