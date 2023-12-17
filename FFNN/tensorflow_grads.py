#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 11:57:46 2023

@author: soumensmacbookair
"""

# Import the libraries
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#%% Forward difference method
def func(theta):
    """theta is a vector"""
    assert theta.size == 3
    f = theta[0]**2 * theta[1] + theta[1]**2 + theta[2]**2
    return f

def func_grad(theta):
    """theta is a vector"""
    assert theta.size == 3
    step_size = 1e-5 * np.linalg.norm(theta, ord=2)
    del_f = np.zeros(shape=(3,))
    unit_vec = np.zeros(shape=(3,))
    for i in range(del_f.size):
        unit_vec[i] = 1 # set
        del_f[i] = (func(theta + step_size * unit_vec) - func(theta))/step_size
        unit_vec[i] = 0 # reset
    return del_f

# print(func_grad(np.array([1,2,3])))

#%% Forward Mode Automatic Differentiation
class Term:
    def __init__(self, value, derivative):
        self.value = value
        self.derivative = derivative

    def __pow__(self, exponent):
        value = self.value ** exponent
        derivative = self.derivative * exponent * (self.value ** (exponent-1))
        return Term(value, derivative)

    def __mul__(self, other):
        value = self.value * other.value
        derivative = self.value * other.derivative + other.value * self.derivative
        return Term(value, derivative)

    def __add__(self, other):
        value = self.value + other.value
        derivative = 1 * other.derivative + 1 * self.derivative
        return Term(value, derivative)

    def __str__(self):
        return f"Value: {self.value}, Derivative: {self.derivative}"

def func_fad(theta, index: int):
    """theta is a vector, index is variable wrt which diff is taken"""
    assert theta.size == 3
    wrt = [1 if i == index else 0 for i in range(theta.size)]

    v1 = Term(theta[0], wrt[0])
    v2 = Term(theta[1], wrt[1])
    v3 = Term(theta[2], wrt[2])

    v = (v1 ** 2) * v2 + v2 ** 2 + v3 ** 2
    return v

print(func_fad(np.array([1,2,3]), 0))
print(func_fad(np.array([1,2,3]), 1))
print(func_fad(np.array([1,2,3]), 2))

#%% Reverse Mode Automatic Differentiation
class Node:

    def __init__(self, value, left, right,
                 left_derivative, right_derivative, index=-1):

        self.value = value
        self.left = left
        self.right = right
        self.left_derivative = left_derivative
        self.right_derivative = right_derivative

        self.adjoint = 0
        self.index = index

        Node.push_stack(self)
        self.parents = []
        self.parents_pos = []

    def __pow__(self, exponent):
        value = self.value ** exponent
        left = self
        right = None
        left_derivative = exponent * (left.value ** (exponent-1))
        right_derivative = 0
        return Node(value, left, right, left_derivative, right_derivative)

    def __mul__(self, other):
        value = self.value * other.value
        left = self
        right = other
        left_derivative = right.value
        right_derivative = left.value
        return Node(value, left, right, left_derivative, right_derivative)

    def __add__(self, other):
        value = self.value + other.value
        left = self
        right = other
        left_derivative = 1
        right_derivative = 1
        return Node(value, left, right, left_derivative, right_derivative)

    def __str__(self):
        return f"""
            node = {self.value}
            left = {self.left.value if self.left != None else None}
            right = {self.right.value if self.right != None else None}
            ld = {self.left_derivative}
            rd = {self.right_derivative}
            adjoint = {self.adjoint}
            index = {self.index}
            parents = {[i.value if i != None else None for i in self.parents]}
            parents_pos = {[i for i in self.parents_pos]}
            """

    @classmethod
    def init_stack(cls):
        cls.stack = []

    @classmethod
    def push_stack(cls, elem):
        cls.stack.append(elem)

def func_rad(theta):
    """theta is a vector"""
    assert theta.size == 3

    Node.init_stack()

    v1 = Node(theta[0], None, None, 0, 0, 0)
    v2 = Node(theta[1], None, None, 0, 0, 1)
    v3 = Node(theta[2], None, None, 0, 0, 2)

    v = (v1 ** 2) * v2 + v2 ** 2 + v3 ** 2

    for elem in reversed(Node.stack):
        if elem.left != None:
            elem.left.parents.append(elem)
            elem.left.parents_pos.append("left")
        if elem.right != None:
            elem.right.parents.append(elem)
            elem.right.parents_pos.append("right")

    for node in reversed(Node.stack):
        if len(node.parents) == 0:
            node.adjoint = 1

        for p, pos in zip(node.parents, node.parents_pos):
            if pos == "left":
                derivative = p.left_derivative
            else:
                derivative = p.right_derivative

            node.adjoint += derivative * p.adjoint

    derivative = [0] * theta.size
    for node in Node.stack:

        if node.left == None and node.right == None:
            derivative[node.index] = node.adjoint

    return v.value, derivative

print(func_rad(np.array([1,2,3])))




