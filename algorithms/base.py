"""
Base module for optimization algorithms.

This module contains the base classes for DCA and BDCA algorithms
with different norms (L1, L2, Linf).
"""

import numpy as np
from typing import List

from config import BaseConfig
from utils import update_mu_exponential


class BaseDCA:
    """
    Base class for DCA (Difference of Convex Algorithm).
    
    This class provides the common structure and methods for DCA algorithms
    with different norms (L1, L2, Linf).
    """
    
    def __init__(
        self,
        num_running_times: int = 1,
        dimentions: List[int] = [],
        num_points: List[int] = [],
        save_to_file: bool = False,
        parameters: dict = None
    ):
        """
        Initialize the DCA algorithm with parameters.
        
        Args:
            parameters: Parameters object for the algorithm
        """
        # Default parameters
        self.config = BaseConfig(num_running_times, dimentions, num_points, save_to_file)

        # Set default
        self.config.parameters.mu = 1.0
        self.config.parameters.muf = 1e-6
        self.config.parameters.delta = 0.5
        self.config.parameters.inner_norm = 1e-6
        self.config.parameters.outer_norm = 1e-6
        
        # Update parameters
        self.config.update_params(parameters, save_log=True)
        # self.config.save_log()

    def run(self, a, x_init, proj_func, ord=1):
        return self._outer_loop(a, x_init, proj_func, ord=1)

    def _inner_loop(self, a, x, mu, proj_func, ord=1):
        """
        Inner loop of the DCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x: numpy array of shape (k, d) representing initial k centers in R^d
            mu: Current smoothing parameter
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of inner iterations
            logs: Dictionary containing logs for this inner loop
        """
        raise NotImplementedError("Subclasses must implement _inner_loop method")
    
    def _outer_loop(self, a, x, proj_func, ord=1):
        """
        Outer loop of the DCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x: numpy array of shape (k, d) representing initial k centers in R^d
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of outer iterations
            iter_logs: Dictionary containing iteration logs
        """
        flag = True
        iter_count = 0
        iter_logs = {}
        
        mu = self.config.parameters.mu
        delta = self.config.parameters.delta
        muf = self.config.parameters.muf

        while flag and mu > muf:
            iter_logs[iter_count] = {'mu': mu}
            x_prev, inner_iter, dca_norms, cost_norms, solution = self._inner_loop(a, x, mu, proj_func, ord)
            
            # Compute norm of the difference
            norm_diff = np.linalg.norm(x_prev - x, 'fro').item()
            
            # Check convergence
            flag = norm_diff >= self.config.parameters.outer_norm
            
            # Update x and mu
            x = x_prev
            # mu = delta * mu
            mu = update_mu_exponential(iter_count, mu, 1e-8, delta)
            
            # Store iteration logs
            iter_logs[iter_count]['inner_iter'] = inner_iter
            iter_logs[iter_count]['outer_norm'] = norm_diff
            iter_logs[iter_count]['dca_norms'] = dca_norms
            iter_logs[iter_count]['cost'] = cost_norms
            iter_logs[iter_count]['solution'] = solution
            iter_count += 1
        for iter_count, values in iter_logs.items():
            self.config.save_log(message=f"INTER COUNT {iter_count}")
            self.config.save_log(values)

        return x, iter_count, iter_logs

    def _compute_w(self, x, a, mu=None):
        """
        Compute the W term for DCA algorithm.
        
        Args:
            x: numpy array of shape (k, d) representing k centers in R^d
            a: numpy array of shape (m, d) representing m data points in R^d
            mu: smoothing parameter (optional, depends on norm)
            
        Returns:
            numpy array of shape (k, d)
        """
        raise NotImplementedError("Subclasses must implement _compute_w method")
    
    def _compute_z(self, x, a, mu):
        """
        Compute the Z term for DCA algorithm.
        
        Args:
            x: numpy array of shape (k, d) representing k centers in R^d
            a: numpy array of shape (m, d) representing m data points in R^d
            mu: smoothing parameter
            
        Returns:
            numpy array of shape (k, d)
        """
        raise NotImplementedError("Subclasses must implement _compute_z method")
    
    def _compute_cost(self, a, x, mu=None):
        """
        Compute the cost function.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x: numpy array of shape (k, d) representing k centers in R^d
            
        Returns:
            float: The cost value
        """
        raise NotImplementedError("Subclasses must implement _compute_cost method")


class BaseBDCA(BaseDCA):
    """
    Base class for BDCA (Boosted Difference of Convex Algorithm).
    
    This class provides the common structure and methods for BDCA algorithms
    with different norms (L1, L2, Linf).
    """
    
    def __init__(self, save_to_file=False, parameters=None):
        """
        Initialize the BDCA algorithm with parameters.
        
        Args:
            parameters: Dictionary of parameters for the algorithm
        """
        # Default parameters
        default_parameters = {
            'mu': 1.0,           # Initial smoothing parameter
            'muf': 1e-6,         # Final smoothing parameter
            'delta': 0.5,        # Reduction factor for mu
            'inner_norm': 1e-6,   # Convergence threshold for inner iterations
            'outer_norm': 1e-6,   # Convergence threshold for outer iterations
            'compute_cost': True, # Flag to compute cost during iterations
            'save_solution': False, # Flag to save solution history
            'alpha': 0.01,         # Alpha parameter for line search
            'beta': 0.5,         # Beta parameter for line search
            'lambda_start': 1.0,  # Initial lambda value
            'gamma': 1.5,        # Gamma parameter for lambda update
            'lambda_history': 10, # Length of lambda history
            'max_search': 20,     # Maximum number of search iterations
            'lambda_min': 1e-3,   # Minimum lambda value
            'lambda_skip': 5     # Number of iterations to skip line search
        }
        
        # Initialize with default parameters
        super().__init__(save_to_file=save_to_file)
        
        # Set default parameters
        self.config.update_params(default_parameters)

        # Update with user-provided parameters
        self.config.update_params(parameters, save_log=True)

    def run(self, a, x_init, proj_func, ord, use_self_adaptive=True):
        """
        Run the BDCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x_init: numpy array of shape (k, d) representing initial k centers in R^d
            use_self_adaptive: Flag to turn on self-adaptivity for lambda in BDCA
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of outer iterations
            iter_logs: Dictionary containing logs for each outer iteration
        """
        return self._outer_loop(a, x_init, proj_func, ord, use_self_adaptive)
    
    def _inner_loop(self, a, x, mu, proj_func, ord, use_self_adaptive):
        """
        Inner loop of the BDCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x: numpy array of shape (k, d) representing initial k centers in R^d
            mu: Current smoothing parameter
            use_self_adaptive: Flag to turn on self-adaptivity for lambda in BDCA
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of inner iterations
            logs: Dictionary containing logs for this inner loop
        """
        raise NotImplementedError("Subclasses must implement _inner_loop method")
    
    def _outer_loop(self, a, x, proj_func, ord, use_self_adaptive):
        """
        Outer loop of the BDCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x: numpy array of shape (k, d) representing initial k centers in R^d
            use_self_adaptive: Flag to turn on self-adaptivity for lambda in BDCA
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of outer iterations
            iter_logs: Dictionary containing iteration logs
        """
        flag = True
        iter_count = 0
        iter_logs = {}
        
        mu = self.config.parameters.mu
        delta = self.config.parameters.delta
        muf = self.config.parameters.muf
        
        while flag and mu > muf:
            iter_logs[iter_count] = {'mu': mu}
            x_prev, inner_iter, dca_norms, cost_norms, search_iters, lambda_iter, solution = self._inner_loop(
                a, x, mu, proj_func, ord, use_self_adaptive
            )

            # Compute norm of the difference
            norm_diff = np.linalg.norm(x_prev - x, 'fro') / (np.linalg.norm(x, 'fro') + 1)
            
            # Check convergence
            flag = norm_diff >= self.config.parameters.outer_norm
            
            # Update x and mu
            x = x_prev
            # mu = delta * mu
            mu = update_mu_exponential(iter_count, mu, 1e-8, delta)
            
            # Store iteration logs
            iter_logs[iter_count]['inner_iter'] = inner_iter
            iter_logs[iter_count]['lambda_iter'] = lambda_iter
            iter_logs[iter_count]['dca_norms'] = dca_norms
            iter_logs[iter_count]['outer_norm'] = norm_diff
            iter_logs[iter_count]['solution'] = solution
            iter_logs[iter_count]['search_iter'] = search_iters
            iter_logs[iter_count]['cost'] = cost_norms
            iter_count += 1

        for iter_count, values in iter_logs.items():
            self.config.save_log(message=f"ITER COUNT {iter_count}")
            self.config.save_log(values)

        return x, iter_count, iter_logs

    # def _compute_cost_pen_opt_unconstrained(self, a, x, mu):
    #     """
    #     Compute the optimized cost penalty function.
        
    #     Args:
    #         a: numpy array of shape (m, d) representing m data points in R^d
    #         x: numpy array of shape (k, d) representing k centers in R^d
    #         mu: smoothing parameter
            
    #     Returns:
    #         float: The cost penalty value
    #     """
    #     raise NotImplementedError("Subclasses must implement _compute_cost_pen_opt_unconstrained method")


class BaseConstrainedDCA(BaseDCA):
    """
    Base class for Constrained DCA (Difference of Convex Algorithm).
    
    This class provides the common structure and methods for Constrained DCA algorithms
    with different norms (L1, L2, Linf).
    """
    
    def __init__(self, save_to_file=False, parameters=None):
        """
        Initialize the Constrained DCA algorithm with parameters.
        
        Args:
            parameters: Dictionary of parameters for the algorithm
        """
        # Default parameters
        default_parameters = {
            'mu': 1.0,           # Initial smoothing parameter
            'muf': 1e-6,         # Final smoothing parameter
            'delta': 0.5,        # Reduction factor for mu
            'inner_norm': 1e-6,   # Convergence threshold for inner iterations
            'outer_norm': 1e-6,   # Convergence threshold for outer iterations
            'compute_cost': True, # Flag to compute cost during iterations
            'save_solution': False, # Flag to save solution history
            'tau': 1.0,          # Projection penalty parameter
            'sigma': 0.1,      # Projection penalty parameter
            'q': 1.0             # Number of constraints per center
        }
        
        # Initialize with default parameters
        super().__init__(save_to_file=save_to_file)

        # Create default params
        self.config.update_params(default_parameters)
        
        # Update with user-provided parameters
        self.config.update_params(parameters, save_log=True)
    
    def run(self, a, x_init, proj_func, ord=1, use_self_adaptive=True):
        """
        Run the Constrained DCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x_init: numpy array of shape (k, d) representing initial k centers in R^d
            proj_func: Function handle for projection operator
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of outer iterations
            iter_logs: Dictionary containing logs for each outer iteration
        """
        mu = self.config.parameters.mu
        muf = self.config.parameters.muf
        delta = self.config.parameters.delta
        
        x = x_init.copy()
        x_old = x.copy()
        
        flag = True
        iter_count = 0
        iter_logs = {}
        
        while flag:
            # Run inner loop
            x , inner_iter, dca_norms, cost_norms, solution = self._inner_loop(a, x, mu, proj_func, ord, use_self_adaptive)
            iter_logs[iter_count] = {"mu": mu}
            # Compute norm of the difference
            norm_diff = np.linalg.norm(x_old - x, 'fro') / (np.linalg.norm(x, 'fro') + 1)
            
            # Store logs
            
            # Update for next iteration
            x_old = x.copy()
            mu = max(mu * delta, muf)

            # Check convergence
            flag = norm_diff >= self.config.parameters.outer_norm and mu > muf
            # Store iteration logs
            iter_logs[iter_count]['inner_iter'] = inner_iter
            iter_logs[iter_count]['outer_norm'] = norm_diff
            iter_logs[iter_count]['dca_norms'] = dca_norms
            iter_logs[iter_count]['cost'] = cost_norms
            iter_logs[iter_count]['solution'] = solution
            iter_count += 1

        for iter_count, values in iter_logs.items():
            self.config.save_log(message=f"INTER COUNT {iter_count}")
            self.config.save_log(values)

        return x, iter_count, iter_logs
    
    def _inner_loop(self, a, x, mu, proj_func, ord, use_self_adaptive=True):
        """
        Inner loop of the Constrained DCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x: numpy array of shape (k, d) representing initial k centers in R^d
            mu: Current smoothing parameter
            proj_func: Function handle for projection operator
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of inner iterations
            logs: Dictionary containing logs for this inner loop
        """
        raise NotImplementedError("Subclasses must implement _inner_loop method")
    
    def _compute_cost_pen_opt(self, a, x, tau, q, mu, proj_func):
        """
        Compute the optimized cost penalty function with constraints.
        
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
        raise NotImplementedError("Subclasses must implement _compute_cost_pen_opt method")


class BaseConstrainedBDCA(BaseBDCA):
    """
    Base class for Constrained BDCA (Boosted Difference of Convex Algorithm).
    
    This class provides the common structure and methods for Constrained BDCA algorithms
    with different norms (L1, L2, Linf).
    """
    
    def __init__(self, parameters=None):
        """
        Initialize the Constrained BDCA algorithm with parameters.
        
        Args:
            parameters: Dictionary of parameters for the algorithm
        """
        # Default parameters
        default_parameters = {
            'mu': 1.0,           # Initial smoothing parameter
            'muf': 1e-6,         # Final smoothing parameter
            'delta': 0.5,        # Reduction factor for mu
            'inner_norm': 1e-6,   # Convergence threshold for inner iterations
            'outer_norm': 1e-6,   # Convergence threshold for outer iterations
            'compute_cost': True, # Flag to compute cost during iterations
            'save_solution': False, # Flag to save solution history
            'alp': 0.01,         # Alpha parameter for line search
            'beta': 0.5,         # Beta parameter for line search
            'lambda_start': 1.0,  # Initial lambda value
            'gamma': 1.5,        # Gamma parameter for lambda update
            'lambda_history': 10, # Length of lambda history
            'max_search': 20,     # Maximum number of search iterations
            'lambda_min': 1e-3,   # Minimum lambda value
            'lambda_skip': 5,    # Number of iterations to skip line search
            'tau': 1.0,          # Projection penalty parameter
            'q': 1.0             # Number of constraints per center
        }
        
        # Initialize with default parameters
        super().__init__(save_to_file=True)
        self.config.update_params(default_parameters)

        # Update with user-provided parameters
        if parameters is not None:
            self.config.update_params(parameters, save_log=True)
    
    def run(self, a, x_init, proj_func, use_self_adaptive=True):
        """
        Run the Constrained BDCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x_init: numpy array of shape (k, d) representing initial k centers in R^d
            proj_func: Function handle for projection operator
            use_self_adaptive: Flag to turn on self-adaptivity for lambda in BDCA
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of outer iterations
            iter_logs: Dictionary containing logs for each outer iteration
        """
        mu = self.config.parameters.mu
        muf = self.config.parameters.muf
        delta = self.config.parameters.delta
        
        x = x_init.copy()
        x_old = x.copy()
        
        flag = True
        iter_count = 0
        iter_logs = {}
        
        while flag:
            # Run inner loop
            x, inner_iter, dca_norms, cost_norms, search_iters, lambda_iter, solution = self._inner_loop(a, x, mu, proj_func, use_self_adaptive)
            iter_logs[iter_count] = {"mu": mu}
            # Compute norm of the difference
            norm_diff = np.linalg.norm(x_old - x, 'fro')
            
            # Store logs
            
            # Update for next iteration
            x_old = x.copy()
            mu = max(mu * delta, muf)
            
            # Check convergence
            flag = norm_diff >= self.config.parameters.outer_norm and mu > muf
            # Store iteration logs
            iter_logs[iter_count]['inner_iter'] = inner_iter
            iter_logs[iter_count]['outer_norm'] = norm_diff
            iter_logs[iter_count]['lambda_iter'] = lambda_iter
            iter_logs[iter_count]['dca_norms'] = dca_norms
            iter_logs[iter_count]['cost'] = cost_norms
            iter_logs[iter_count]['solution'] = solution
            iter_logs[iter_count]['search_iter'] = search_iters
            iter_count += 1
        
        for iter_count, values in iter_logs.items():
            self.config.save_log(message=f"INTER COUNT {iter_count}")
            self.config.save_log(values)

        return x, iter_count, iter_logs
    
    def _inner_loop(self, a, x, mu, proj_func, use_self_adaptive):
        """
        Inner loop of the Constrained BDCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x: numpy array of shape (k, d) representing initial k centers in R^d
            mu: Current smoothing parameter
            proj_func: Function handle for projection operator
            use_self_adaptive: Flag to turn on self-adaptivity for lambda in BDCA
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of inner iterations
            logs: Dictionary containing logs for this inner loop
        """
        raise NotImplementedError("Subclasses must implement _inner_loop method")
    
    def _compute_cost_pen_opt(self, a, x, tau, q, mu, proj_func):
        """
        Compute the optimized cost penalty function with constraints.
        
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
        raise NotImplementedError("Subclasses must implement _compute_cost_pen_opt method")


class BaseConstrainedBDCAV2(BaseBDCA):
    def run(self, a, x_init, proj_func, use_self_adaptive=True):
        """
        Run the Constrained BDCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x_init: numpy array of shape (k, d) representing initial k centers in R^d
            proj_func: Function handle for projection operator
            use_self_adaptive: Flag to turn on self-adaptivity for lambda in BDCA
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of outer iterations
            iter_logs: Dictionary containing logs for each outer iteration
        """
        mu = self.config.parameters.mu
        muf = self.config.parameters.muf
        delta = self.config.parameters.delta
        tau = self.config.parameters.tau
        sigma = self.config.parameters.sigma
        
        x = x_init.copy()
        x_old = x.copy()
        
        flag = True
        iter_count = 0
        iter_logs = {}
        
        while flag:
            # Run inner loop
            x, inner_iter, dca_norms, cost_norms, search_iters, lambda_iter, solution = self._inner_loop(a, x, mu, proj_func, use_self_adaptive)
            iter_logs[iter_count] = {"mu": mu}
            # Compute norm of the difference
            norm_diff = np.linalg.norm(x_old - x, 'fro')
            
            # Store logs
            
            # Update for next iteration
            x_old = x.copy()
            mu = max(mu * delta, muf)
            self.config.parameters.tau = sigma * tau
            # Check convergence
            flag = norm_diff >= self.config.parameters.outer_norm and mu > muf
            # Store iteration logs
            iter_logs[iter_count]['inner_iter'] = inner_iter
            iter_logs[iter_count]['outer_norm'] = norm_diff
            iter_logs[iter_count]['lambda_iter'] = lambda_iter
            iter_logs[iter_count]['dca_norms'] = dca_norms
            iter_logs[iter_count]['cost'] = cost_norms
            iter_logs[iter_count]['solution'] = solution
            iter_logs[iter_count]['search_iter'] = search_iters
            iter_count += 1
        
        for iter_count, values in iter_logs.items():
            self.config.save_log(message=f"INTER COUNT {iter_count}")
            self.config.save_log(values)

        return x, iter_count, iter_logs