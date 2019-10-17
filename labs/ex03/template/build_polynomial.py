# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    fn = lambda a: [pow(a, d) for d in range(0, degree + 1)]
    result = map(fn, x)
    return np.asarray(list(result))
