"""
Cost functions for optimization algorithms.

This module contains cost functions for different norms (L1, L2, Linf).
"""

import numpy as np
from utils import project_onto_l1_ball


def cost(a, x, ord):
    x_reshaped = x[:, None, :]
    a_reshaped = a[None, :, :]
    norm = np.linalg.norm(x_reshaped - a_reshaped, ord=ord, axis=2)
    indices = np.argmin(norm, axis=0)
    min_cost = np.min(norm, axis=0)
    cost = np.sum(min_cost).item()
    return cost, indices


def cost_pen_opt(a, x, mu, ord, proj_func):
    x_reshaped = x[:, None, :]
    a_shaped = a[None, :, :]
    temp = (x_reshaped - a_shaped) / mu
    norm = np.linalg.norm(temp, ord=ord, axis=2)
    temp_reshaped = temp.reshape(-1, a.shape[1])
    projected = proj_func(temp_reshaped)
    diff = temp_reshaped - projected
    diff_norm = np.linalg.norm(diff, axis=1) ** 2
    diff_norm = diff_norm.reshape(x.shape[0], a.shape[0])
    cost = mu / 2 * np.sum(np.minimum(norm ** 2, diff_norm), axis=1).sum()
    return cost

def cost_l1(a, x):
    """
    Compute the L1 cost function.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
        
    Returns:
        float: The cost value
        numpy array: The indices of the closest centers for each data point
    """

    x_reshaped = x[:, None, :]
    a_reshaped = a[None, :, :]
    norms = np.sum(np.abs(x_reshaped - a_reshaped), axis=2)
    indices = np.argmin(norms, axis=0)
    min_cost = np.min(norms, axis=0)
    return np.sum(min_cost).item(), indices


def cost_l2(a, x):
    """
    Compute the L2 cost function.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
    
    Returns:
        tuple: (cost, indices)
            cost: The L2 cost value
            indices: numpy array of shape (m,) representing the indices of the closest centers for each data point
    """
    m = a.shape[0]
    k = x.shape[0]
    d = a.shape[1]  # Thêm dòng này để định nghĩa biến d
    
    # Compute the L2 norms between x_i and a_j
    x_a_norm = np.linalg.norm(
        (np.reshape(x, (k, 1, d)) - np.reshape(a, (1, m, d))),
        ord=2,
        axis=2
    )
    
    # Find the closest center for each data point
    indices = np.argmin(x_a_norm, axis=0)
    
    # Compute the cost
    cost = 0
    for i in range(m):
        cost += np.linalg.norm(a[i] - x[indices[i]], ord=2)
    
    return cost, indices

def cost_linf(a, x, mu=None):
    """
    Compute the Linf cost function.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
    
    Returns:
        tuple: (cost, indices)
            cost: The Linf cost value
            indices: numpy array of shape (m,) representing the indices of the closest centers for each data point
    """
    x_reshaped = x[:, None, :]
    a_reshaped = a[None, :, :]
    
    diff = x_reshaped - a_reshaped
    norm = np.max(np.abs(diff), axis=2)
    cost = np.sum(np.min(norm, axis=0))
    indices = np.argmin(norm, axis=0)

    return cost, indices


def cost_pen_opt_unconstrained_l1(a, x, mu):
    """
    Compute the optimized L1 cost penalty function without constraints.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
        mu: smoothing parameter
    
    Returns:
        float: The cost penalty value
    """
    x_reshaped = x[:, None, :]
    a_reshaped = a[None, :, :]
    diff = (x_reshaped - a_reshaped) / 2
    norm = np.sqrt(np.sum(diff ** 2, axis=2))
    distances = diff - np.sign(diff)
    distances[diff < 1] = 0
    dis_norm = np.sqrt(np.sum(distances ** 2, axis=2))
    cost = mu / 2 * np.sum(np.minimum(norm ** 2 - dis_norm ** 2, np.zeros_like(norm)), axis=0)
    cost = np.sum(cost)
    return cost


def cost_pen_opt_unconstrained_l2(a, x, mu):
    """
    Compute the optimized L2 cost penalty function without constraints.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
        mu: smoothing parameter
    
    Returns:
        float: The cost penalty value
    """
    m = a.shape[0]
    k = x.shape[0]
    d = a.shape[1]  # Thêm dòng này để định nghĩa biến d
    
    # Compute the L2 cost
    cost_val, _ = cost_l2(a, x)
    
    # Compute the penalty term
    penalty = 0
    
    # All differences of rows between X - A in the format k*m x d
    temp = (np.reshape(x, (k, 1, d)) - np.reshape(a, (1, m, d))) / mu
    
    # Compute the norm of each difference
    temp_norm = np.linalg.norm(temp, ord=2, axis=2)
    
    # Compute the penalty
    penalty_terms = np.maximum(0, temp_norm - 1)**2
    penalty = 0.5 * mu * np.sum(penalty_terms)
    
    return cost_val + penalty


def cost_pen_opt_unconstrained_linf(a, x, mu):
    """
    Compute the optimized Linf cost penalty function without constraints.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
        mu: smoothing parameter
    
    Returns:
        float: The cost penalty value
    """
    m = a.shape[0]
    k = x.shape[0]
    d = a.shape[1]  # Thêm dòng này để định nghĩa biến d
    
    x_reshaped = x[:, None, :]
    a_reshaped = a[None, :, :]
    temp = (x_reshaped - a_reshaped) / mu
    
    norm = np.linalg.norm(temp, axis=2)
    temp_reshaped = temp.reshape(-1, d)
    proj_l1 = project_onto_l1_ball(temp_reshaped)
    diff = temp_reshaped - proj_l1
    diff_norm = np.linalg.norm(diff, axis=1) ** 2
    diff_norm = diff_norm.reshape(k, m)
    # c = mu/2 * sum(min(XAnorm^2 - disnorm^2))
    cost = mu / 2 * np.sum(np.minimum(norm**2, diff_norm), axis=1).sum()

    return cost


def cost_pen_opt_l1(a, x, tau, q, mu, proj_func):
    """
    Compute the optimized L1 cost penalty function with constraints.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
        tau: Projection penalty parameter
        q: Number of constraints per center
        mu: smoothing parameter
        proj_func: Function handle for projection operator
    
    Returns:
        float: The cost penalty value
    """
    # Compute the unconstrained cost penalty
    cost_pen = cost_pen_opt_unconstrained_l1(a, x, mu)
    
    # Compute the constraint penalty
    u = proj_func(x)
    constraint_penalty = 0.5 * tau * np.linalg.norm(x - u, 'fro')**2
    
    return cost_pen + constraint_penalty


def cost_pen_opt_l2(a, x, tau, q, mu, proj_func):
    """
    Compute the optimized L2 cost penalty function with constraints.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
        tau: Projection penalty parameter
        q: Number of constraints per center
        mu: smoothing parameter
        proj_func: Function handle for projection operator
    
    Returns:
        float: The cost penalty value
    """
    # Compute the unconstrained cost penalty
    cost_pen = cost_pen_opt_unconstrained_l2(a, x, mu)
    
    # Compute the constraint penalty
    u = proj_func(x)
    constraint_penalty = 0.5 * tau * np.linalg.norm(x - u, 'fro')**2
    
    return cost_pen + constraint_penalty


def cost_pen_opt_linf(a, x, tau, q, mu, proj_func):
    """
    Compute the optimized Linf cost penalty function with constraints.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
        tau: Projection penalty parameter
        q: Number of constraints per center
        mu: smoothing parameter
        proj_func: Function handle for projection operator
    
    Returns:
        float: The cost penalty value
    """
    # Compute the unconstrained cost penalty
    cost_pen = cost_pen_opt_unconstrained_linf(a, x, mu)
    
    # Compute the constraint penalty
    u = proj_func(x)
    constraint_penalty = 0.5 * tau * np.linalg.norm(x - u, 'fro')**2
    
    return cost_pen + constraint_penalty
