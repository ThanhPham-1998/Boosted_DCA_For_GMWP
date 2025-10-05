"""
Visualization module for optimization algorithms.

This module contains functions for visualizing optimization results,
including clustering, convergence, and lambda search.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
mpl.rcParams['animation.codec'] = 'mpeg4'


def generate_random_colors(n):
    """Tạo n màu ngẫu nhiên ở dạng hex"""
    colors = []
    for _ in range(n):
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        colors.append(color)
    return colors


def plot_clusters(a, x, indices=None, colors=None):
    """
    Plot data points and cluster centers.
    
    Args:
        a: numpy array of shape (m, d) representing m data points in R^d
        x: numpy array of shape (k, d) representing k centers in R^d
        indices: numpy array of shape (m,) representing the indices of the closest centers for each data point
                (if None, will be computed)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Check dimensions
    if a.shape[1] != 2 and a.shape[1] != 3:
        raise ValueError("This function only supports 2D or 3D data")
    
    # Compute indices if not provided
    if indices is None:
        m = a.shape[0]
        k = x.shape[0]
        d = a.shape[1]
        
        # Compute distances between x_i and a_j
        distances = np.zeros((k, m))
        for i in range(k):
            for j in range(m):
                distances[i, j] = np.linalg.norm(x[i] - a[j])
        
        # Find the closest center for each data point
        indices = np.argmin(distances, axis=0)
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    if colors is None:
        colors = generate_random_colors(x.shape[0])
    
    # 2D or 3D plot
    if a.shape[1] == 2:
        ax = fig.add_subplot(111)
        
        # Plot data points
        for i in range(x.shape[0]):
            cluster_points = a[indices == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=20, alpha=0.5, c=colors[i], label=f'Cluster {i+1}')
        
        # Plot centers
        ax.scatter(x[:, 0], x[:, 1], c=colors or 'red', marker='x', s=100, linewidths=3, label='Centers')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
    else:  # 3D
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot data points
        for i in range(x.shape[0]):
            cluster_points = a[indices == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], s=20, alpha=0.5, c=colors[i], label=f'Cluster {i+1}')
        
        # Plot centers
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors or 'red', marker='x', s=100, linewidths=3, label='Centers')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    
    return fig


def plot_convergence(iter_logs, norm_type=1):
    """
    Plot convergence of the optimization algorithm.
    
    Args:
        iter_logs: Dictionary containing logs for each outer iteration
        norm_type: Type of norm used ('L1', 'L2', or 'Linf')
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot outer iteration norms
    ax1 = fig.add_subplot(221)
    outer_norms = [iter_logs[i]['outer_norm'] for i in range(len(iter_logs))]
    ax1.plot(outer_norms, 'o-', linewidth=2)
    ax1.set_xlabel('Outer Iteration')
    ax1.set_ylabel('Norm Difference')
    ax1.set_title('Outer Iteration Convergence')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # Plot mu values
    ax2 = fig.add_subplot(222)
    mu_values = [iter_logs[i]['mu'] for i in range(len(iter_logs))]
    ax2.plot(mu_values, 'o-', linewidth=2)
    ax2.set_xlabel('Outer Iteration')
    ax2.set_ylabel('Mu Value')
    ax2.set_title('Mu Values')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    # Plot inner iteration counts
    ax3 = fig.add_subplot(223)
    inner_iters = [iter_logs[i]['inner_iter'] for i in range(len(iter_logs))]
    ax3.plot(inner_iters, 'o-', linewidth=2)
    ax3.set_xlabel('Outer Iteration')
    ax3.set_ylabel('Inner Iterations')
    ax3.set_title('Inner Iteration Counts')
    ax3.grid(True)
    
    # Plot cost values if available
    ax4 = fig.add_subplot(224)
    if 'cost' in iter_logs[0]:
        for i in range(len(iter_logs)):
            costs = iter_logs[i]['cost']
            ax4.plot(costs, label=f'Outer Iter {i}')
        ax4.set_xlabel('Inner Iteration')
        ax4.set_ylabel(f'L{norm_type} Cost')
        ax4.set_title('Cost Values')
        ax4.set_yscale('log')
        ax4.grid(True)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Cost values not available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax4.transAxes)
    
    plt.tight_layout()
    return fig


def plot_lambda_search(iter_logs):
    """
    Plot lambda search for BDCA algorithm.
    
    Args:
        iter_logs: Dictionary containing logs for each outer iteration
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Check if lambda values are available
    if 'lambda_iter' not in iter_logs[0]:
        raise ValueError("Lambda values not available in logs")
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot lambda values
    ax1 = fig.add_subplot(221)
    for i in range(len(iter_logs)):
        lambda_values = iter_logs[i]['lambda_iter']
        ax1.plot(lambda_values, label=f'Outer Iter {i}')
    ax1.set_xlabel('Inner Iteration')
    ax1.set_ylabel('Lambda Value')
    ax1.set_title('Lambda Values')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()
    
    # Plot search iterations
    ax2 = fig.add_subplot(222)
    for i in range(len(iter_logs)):
        search_iters = iter_logs[i]['search_iter']
        ax2.plot(search_iters, label=f'Outer Iter {i}')
    ax2.set_xlabel('Inner Iteration')
    ax2.set_ylabel('Search Iterations')
    ax2.set_title('Search Iterations')
    ax2.grid(True)
    ax2.legend()
    
    # Plot histogram of lambda values
    ax3 = fig.add_subplot(223)
    all_lambdas = []
    for i in range(len(iter_logs)):
        all_lambdas.extend(iter_logs[i]['lambda_iter'])
    ax3.hist(all_lambdas, bins=30)
    ax3.set_xlabel('Lambda Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Lambda Value Distribution')
    ax3.set_xscale('log')
    ax3.grid(True)
    
    # Plot histogram of search iterations
    ax4 = fig.add_subplot(224)
    all_searches = []
    for i in range(len(iter_logs)):
        all_searches.extend(iter_logs[i]['search_iter'])
    ax4.hist(all_searches, bins=max(10, int(max(all_searches)) + 1))
    ax4.set_xlabel('Search Iterations')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Search Iteration Distribution')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig


def plot_no_skip_vs_skip_lambda(iter_logs_no_skip, iter_logs_skip, skip_value):
    """
    Compare lambda values with and without skipping.
    
    Args:
        iter_logs_no_skip: Dictionary containing logs for each outer iteration without skipping
        iter_logs_skip: Dictionary containing logs for each outer iteration with skipping
        skip_value: Number of iterations to skip
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Check if lambda values are available
    if 'lambda_iter' not in iter_logs_no_skip[0] or 'lambda_iter' not in iter_logs_skip[0]:
        raise ValueError("Lambda values not available in logs")
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot lambda values
    ax1 = fig.add_subplot(221)
    for i in range(len(iter_logs_no_skip)):
        lambda_values = iter_logs_no_skip[i]['lambda_iter']
        ax1.plot(lambda_values, label=f'Outer Iter {i}')
    ax1.set_xlabel('Inner Iteration')
    ax1.set_ylabel('Lambda Value (No Skip)')
    ax1.set_title('Lambda Values (No Skip)')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()
    
    ax2 = fig.add_subplot(222)
    for i in range(len(iter_logs_skip)):
        lambda_values = iter_logs_skip[i]['lambda_iter']
        ax2.plot(lambda_values, label=f'Outer Iter {i}')
    ax2.set_xlabel('Inner Iteration')
    ax2.set_ylabel(f'Lambda Value (Skip {skip_value})')
    ax2.set_title(f'Lambda Values (Skip {skip_value})')
    ax2.set_yscale('log')
    ax2.grid(True)
    ax2.legend()
    
    # Plot search iterations
    ax3 = fig.add_subplot(223)
    for i in range(len(iter_logs_no_skip)):
        search_iters = iter_logs_no_skip[i]['search_iter']
        ax3.plot(search_iters, label=f'Outer Iter {i}')
    ax3.set_xlabel('Inner Iteration')
    ax3.set_ylabel('Search Iterations (No Skip)')
    ax3.set_title('Search Iterations (No Skip)')
    ax3.grid(True)
    ax3.legend()
    
    ax4 = fig.add_subplot(224)
    for i in range(len(iter_logs_skip)):
        search_iters = iter_logs_skip[i]['search_iter']
        ax4.plot(search_iters, label=f'Outer Iter {i}')
    ax4.set_xlabel('Inner Iteration')
    ax4.set_ylabel(f'Search Iterations (Skip {skip_value})')
    ax4.set_title(f'Search Iterations (Skip {skip_value})')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    return fig


def generate_solution_movie(iter_logs, a, save_path, colors=None, num_clusters=1, fps=5):
    m, d = a.shape
    if d != 2 and d != 3:
        raise ValueError("This function only supports 2D or 3D data")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    if d == 3:
        ax = fig.add_subplot(111, projection='3d')
    colors = generate_random_colors(num_clusters) if not colors else colors

    def init():
        ax.clear()
        if d == 2:
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(-0.5, 2.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-5, 5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        ax.set_title('Clustering Animation')
        ax.grid(True)
        return []
    
    def update(frame):
        ax.clear()
        mu_idx, inner_iter = frame
        log = iter_logs[mu_idx]
        I = log['solution'][inner_iter]["indices"]  # Chỉ số cụm
        x = log['solution'][inner_iter]["centers"]  # Chỉ số cụm
        if d == 2:
            for i in range(num_clusters):
                mask = I == i
                ax.scatter(a[mask, 0], a[mask, 1], c=[colors[i]], label=f'Cluster {i}', s=20)
            ax.scatter(x[:, 0], x[:, 1], c=colors, marker="x", s=100, label="Center")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        else:
            for i in range(num_clusters):
                mask = I == i
                ax.scatter(a[mask, 0], a[mask, 1], a[mask, 2], c=[colors[i]], label=f'Cluster {i}', s=20)
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors, marker="x", s=100, label="Center")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        ax.legend()
        ax.grid(True)
        return []

    frames = []
    for i in iter_logs:
        inner_iter_count = len(iter_logs[i]["solution"])
        for inner_iter in range(inner_iter_count):
            frames.append((i, inner_iter))

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    if save_path:
        writer = FFMpegWriter(fps=50, bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(save_path, writer=writer)
        print(f"Video saved at {save_path}")
    else:
        plt.show()
    plt.close()
