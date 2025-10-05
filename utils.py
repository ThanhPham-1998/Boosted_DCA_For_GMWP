"""
Utility functions for optimization algorithms.

This module contains utility functions for optimization algorithms,
including projection operators and file handling.
"""

import numpy as np
import os
import datetime
import functools
from time import time


def project_onto_l1_ball(x, radius=1.0):
    """
    Project points onto the L1 ball.
    
    Args:
        x: numpy array of shape (n, d) representing n points in R^d
        radius: radius of the L1 ball (default: 1.0)
    
    Returns:
        numpy array of shape (n, d) representing the projected points
    """
    # Handle the case where x is a vector
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n, d = x.shape
    
    # Initialize output
    y = np.zeros_like(x)
    
    # For each point
    for i in range(n):
        # Get the point
        v = x[i, :]
        
        # Compute L1 norm
        v_norm = np.sum(np.abs(v))
        
        # If already inside the ball, no projection needed
        if v_norm <= radius:
            y[i, :] = v
        else:
            # Sort absolute values in descending order
            u = np.sort(np.abs(v))[::-1]
            
            # Compute the index k
            sv = np.cumsum(u)
            rho = np.nonzero(u * np.arange(1, d + 1) > sv - radius)[0][-1]
            
            # Compute the Lagrange multiplier
            theta = (sv[rho] - radius) / (rho + 1)
            
            # Compute the projection
            y[i, :] = np.sign(v) * np.maximum(np.abs(v) - theta, 0)

    return y


def project_onto_l2_ball(x, radius=1.0):
    """
    Project points onto the L2 ball.
    
    Args:
        x: numpy array of shape (n, d) representing n points in R^d
        radius: radius of the L2 ball (default: 1.0)
    
    Returns:
        numpy array of shape (n, d) representing the projected points
    """
    # Handle the case where x is a vector
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n, d = x.shape
    
    # Initialize output
    y = np.zeros_like(x)
    
    # For each point
    for i in range(n):
        # Get the point
        v = x[i, :]
        
        # Compute L2 norm
        v_norm = np.linalg.norm(v, 2)
        
        # If already inside the ball, no projection needed
        if v_norm <= radius:
            y[i, :] = v
        else:
            # Scale to the boundary of the ball
            y[i, :] = v * radius / v_norm
    
    return y


def project_onto_linf_ball(x, radius=1.0):
    """
    Project points onto the L-infinity ball.
    
    Args:
        x: numpy array of shape (n, d) representing n points in R^d
        radius: radius of the L-infinity ball (default: 1.0)
    
    Returns:
        numpy array of shape (n, d) representing the projected points
    """
    # Handle the case where x is a vector
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    projected = x - np.sign(x)
    projected[np.abs(x) < radius] = 0

    return projected
        

def project_onto_affine(x, A, b):
    """
    Project points onto the affine subspace defined by Ax = b.
    
    Args:
        x: numpy array of shape (n, d) representing n points in R^d
        A: numpy array of shape (m, d) representing the coefficient matrix
        b: numpy array of shape (m,) representing the right-hand side
    
    Returns:
        numpy array of shape (n, d) representing the projected points
    """
    # Handle the case where x is a vector
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n, d = x.shape
    m = A.shape[0]
    
    # Initialize output
    y = np.zeros_like(x)
    
    # For each point
    for i in range(n):
        # Get the point
        v = x[i, :]
        
        # Compute the projection
        # y = x - A^T (AA^T)^(-1) (Ax - b)
        AAT = A @ A.T
        if m == 1:  # Special case for a single constraint
            AAT = AAT.reshape(1, 1)
            AAT_inv = 1.0 / AAT
        else:
            AAT_inv = np.linalg.inv(AAT)
        
        y[i, :] = v - A.T @ AAT_inv @ (A @ v - b)
    
    return y


def project_onto_polygon(x, vertices):
    """
    Project points onto a convex polygon defined by its vertices.
    
    Args:
        x: numpy array of shape (n, 2) representing n points in R^2
        vertices: numpy array of shape (v, 2) representing the vertices of the polygon
    
    Returns:
        numpy array of shape (n, 2) representing the projected points
    """
    # Handle the case where x is a vector
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n, d = x.shape
    if d != 2:
        raise ValueError("This function only supports 2D points")
    
    # Initialize output
    y = np.zeros_like(x)
    
    # For each point
    for i in range(n):
        # Get the point
        p = x[i, :]
        
        # Check if the point is inside the polygon
        if is_point_in_polygon(p, vertices):
            y[i, :] = p
        else:
            # Find the closest point on the polygon boundary
            min_dist = float('inf')
            closest_point = None
            
            # Check each edge of the polygon
            for j in range(len(vertices)):
                v1 = vertices[j]
                v2 = vertices[(j + 1) % len(vertices)]
                
                # Project onto the line segment
                proj = project_onto_line_segment(p, v1, v2)
                
                # Compute distance
                dist = np.linalg.norm(p - proj)
                
                # Update if closer
                if dist < min_dist:
                    min_dist = dist
                    closest_point = proj
            
            y[i, :] = closest_point
    
    return y


def is_point_in_polygon(point, vertices):
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        point: numpy array of shape (2,) representing a point in R^2
        vertices: numpy array of shape (v, 2) representing the vertices of the polygon
    
    Returns:
        bool: True if the point is inside the polygon, False otherwise
    """
    x, y = point
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def project_onto_line_segment(point, v1, v2):
    """
    Project a point onto a line segment.
    
    Args:
        point: numpy array of shape (2,) representing a point in R^2
        v1: numpy array of shape (2,) representing the first endpoint of the line segment
        v2: numpy array of shape (2,) representing the second endpoint of the line segment
    
    Returns:
        numpy array of shape (2,) representing the projected point
    """
    # Convert to numpy arrays
    point = np.array(point)
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Vector from v1 to v2
    line_vec = v2 - v1
    
    # Vector from v1 to point
    point_vec = point - v1
    
    # Compute the projection
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec.dot(line_unitvec)
    
    # Ensure the projection is on the line segment
    if point_vec_scaled < 0:
        return v1
    elif point_vec_scaled > line_len:
        return v2
    else:
        return v1 + line_unitvec * point_vec_scaled


def create_unique_filename(base_name, extension):
    """
    Create a unique filename by appending a timestamp.
    
    Args:
        base_name: Base name for the file
        extension: File extension (without the dot)
    
    Returns:
        str: Unique filename
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def save_results(x, iter_logs, base_name="results", save_dir=None):
    """
    Save optimization results to files.
    
    Args:
        x: numpy array of shape (k, d) representing optimized centers
        iter_logs: Dictionary containing logs for each outer iteration
        base_name: Base name for the files
        save_dir: Directory to save the files (default: current directory)
    
    Returns:
        dict: Dictionary with paths to saved files
    """
    if save_dir is None:
        save_dir = os.getcwd()
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save solution
    solution_file = os.path.join(save_dir, create_unique_filename(f"{base_name}_solution", "npy"))
    np.save(solution_file, x)
    
    # Save logs
    logs_file = os.path.join(save_dir, create_unique_filename(f"{base_name}_logs", "npy"))
    np.save(logs_file, iter_logs)
    
    return {
        "solution": solution_file,
        "logs": logs_file
    }


def update_mu(mu, iteration, cost_new, cost_old=None, mu_min=1e-6, alpha=0.01, threshold=0.01, const=0.9):
    """
    Update mu using a hybrid adaptive schedule based on iteration and cost reduction.

    Parameters:
    mu (float): Current value of mu
    iteration (int): Current iteration number
    cost_new (float): Current cost (from cost_pen_opt_unconstrained_Linf or cost_Linf)
    cost_old (float, optional): Previous cost (default: None, skips adaptive adjustment)
    mu_0 (float): Initial mu value (default: 1.0)
    mu_min (float): Minimum allowed mu value (default: 1e-6)
    alpha (float): Decay rate for exponential schedule (default: 0.01)
    threshold (float): Threshold for relative cost change to trigger adaptive reduction (default: 0.01)
    const (float): Multiplicative factor for adaptive reduction (default: 0.9)

    Returns:
    float: Updated mu value
    """
    # Base exponential decay
    mu_new = mu * np.exp(-alpha * iteration)

    # Adaptive adjustment based on cost reduction
    if cost_old is not None and cost_old != 0:
        rel_change = abs(cost_new - cost_old) / abs(cost_old)
        if rel_change < threshold:
            # If cost reduction is small, reduce mu more aggressively
            mu_new *= const

    # Enforce minimum mu
    mu_new = max(mu_new, mu_min)

    return mu_new

# Alternative 1: Simple Exponential Decay
def update_mu_exponential(iteration, mu_0=1.0, mu_min=1e-6, alpha=0.01):
    """
    Update mu using simple exponential decay.

    Parameters:
    iteration (int): Current iteration number
    mu_0 (float): Initial mu value (default: 1.0)
    mu_min (float): Minimum allowed mu value (default: 1e-6)
    alpha (float): Decay rate (default: 0.01)

    Returns:
    float: Updated mu value
    """
    return max(mu_0 * np.exp(-alpha * (iteration + 1)), mu_min)

# Alternative 2: Adaptive Decrease Based on Cost
def update_mu_adaptive(mu, cost_new, cost_old, mu_min=1e-6, threshold=0.01, const=0.9):
    """
    Update mu based on relative cost change.

    Parameters:
    mu (float): Current mu value
    cost_new (float): Current cost
    cost_old (float): Previous cost
    mu_min (float): Minimum allowed mu value (default: 1e-6)
    threshold (float): Threshold for relative cost change (default: 0.01)
    const (float): Multiplicative factor for reduction (default: 0.9)

    Returns:
    float: Updated mu value
    """
    if cost_old != 0:
        rel_change = abs(cost_new - cost_old) / abs(cost_old)
        if rel_change < threshold:
            mu *= const
    return max(mu, mu_min)


def profiler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper
