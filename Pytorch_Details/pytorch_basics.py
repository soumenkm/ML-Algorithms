#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 11:51:15 2024

@author: soumensmacbookair
"""

#%% Imports
import numpy as np
import torch

#%% Create tensor
a = torch.rand(size=(2,3), dtype=torch.float64)
b = torch.tensor(data=[[1,2,3],[4,5,6]],
                 dtype=torch.float32,
                 device=None,
                 requires_grad=False)

"""
other ways to create a tensor:
    zeros, zeros_like, ones, ones_like, arange, range, linspace, full, full_like,
    eye, from_numpy, rand, rand_like, randint, randint_like, randn, randn_like
"""

# Attributes and methods of tensor
attr = [i for i in a.__dir__() if not i.startswith("_")]

# Cast to different type (2 ways)
a = a.to(torch.int64)
a = a.type(torch.int32)

# Cast to numpy or list
a = torch.tensor([[1,2,3],[4,5,6]])
a = a.numpy()
a = a.tolist()

#%% Math operations on tensor
# Pointwise or Elementwise operation
"""
torch.operation()
operation is common math operations like:
    abs, add (+), ceil, clip, cos, div (/), exp, floor, log, log10, log2,
    logical_and, logical_or, logical_not, logical_xor, mul (*), neg (-a),
    pow (**), round, sign, sin, softmax, square, sqrt, sub (-), tan
"""

a = torch.round(torch.rand(size=(3,3)))
b = torch.round(torch.rand(size=(3,3)))
c = torch.logical_and(a, b)
d = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
e = torch.softmax(d, dim=1)

# Reduction operation (reduces to rank along a dimension)
"""
torch.operation()
operation is common math reduction operations like:
    argmax, argmin, max, min, all, any, mean, median, mode, norm,
    prod, sum, quantile, std, var, count_nonzero
"""

a = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
b = torch.argmax(a, dim=-1, keepdims=True)
c = torch.tensor([[1,0,1],[1,1,0]], dtype=torch.bool)
d = torch.all(c, dim=-1) # if all of the elements are True then return True along a dim

# Comparison operation
"""
torch.operation()
operation is common comparison operations like:
    eq (==), ge (>=), gt (>), ne (!=), le (<=), lt (<), isin, isinf,
    isposinf, isneginf, isnan, maximum, minimum, sort, argsort, allclose
"""

a = torch.tensor([[1,3,2],[2,1,2],[3,2,1],[2,1,1]], dtype=torch.float32)
b = (a == 1) # broadcasted for elementwise comparison
c = torch.isin(a, torch.tensor([1,2,3,4,5,6,7,8])) # test is each element of a is in the list
d = torch.maximum(a, b.type(torch.float64)) # test element wise maximum betweeon a and b

# Linear Algebra operations
"""
torch.operation()
operation is common LA operations like:
    dot or inner (for 1d only), inverse, det, matmul, mv (mat, vec mult), outer,
    qr, svd etc. For more use torch.linalg API
"""

a = torch.tensor([[1,2,3],[4,5,6]]) # dim 2
b = torch.tensor([1,2,3]) # dim 1
c = torch.mv(a, b) # dim 1

#%% Tensor indexing and manipulation
"""
torch.operation()
operation is common indexing operations like:
    [] operator, concat, reshape, transpose, squeeze, unsqueeze,
    stack, split, where
"""

a = torch.tensor([[1,2,3],[4,5,6]]) # (2,3)
b = torch.tensor([[7,8,9],[10,11,12]]) # (2,3)

c = torch.concat([a, b], dim=0) # (4,3) [2+2 = 4 at dim 0]
d = torch.concat([a, b], dim=1) # (2,6) [3+3 = 6 at dim 1]

a = torch.tensor([[[1,2,3],[4,5,6]]]) # (1,2,3)
b = torch.squeeze(a) # (2,3)

a = torch.tensor([[1,2,3],[4,5,6]]) # (2,3)
b = torch.unsqueeze(a, dim=0) # (1,2,3)

a = torch.tensor([[1,2,3],[4,5,6]]) # (2,3)
b = torch.tensor([[7,8,9],[10,11,12]]) # (2,3)
c = torch.stack([a,b], dim=0) # (2, 2, 3) [2 elements in dim 0 added - new dim]
c = torch.stack([a,b], dim=1) # (2, 2, 3) [2 elements in dim 1 added - new dim]
c = torch.stack([a,b], dim=2) # (2, 3, 2) [2 elements in dim 2 added - new dim]

a = torch.tensor([[1,2,3],[4,5,6]]) # (2,3)
b = torch.split(a, split_size_or_sections=1, dim=0) # opposite of concat, along dim 0, split based on 1 elements

a = torch.tensor([[1,2,3],[4,5,6]]) # (2,3)
b = torch.where(a <= 2, 1, 0) # (broadcasted) same as elementwise ?:: operator of C++

#%% Dataset and Dataloaders











