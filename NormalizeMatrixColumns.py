# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:09:34 2020

@author: gmoha
"""

import numpy as np

def normalizeColumns(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord = 2, axis =0, keepdims = True)
    x_norm = np.where(x_norm == 0, 1, x_norm)
    
    # Divide x by its norm.
    x = x/x_norm
    ### END CODE HERE ###

    return (x,x_norm)