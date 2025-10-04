import numpy as np

from .base import BaseDCA
from cost import cost



class DCA(BaseDCA):
    """
    DCA (Difference of Convex Algorithm) for Linf optimization.
    
    This class implements the DCA algorithm for solving Linf optimization problems.
    """
    
    def _inner_loop(self, a, x, mu, proj_func=None, ord=1):
        """
        Inner loop of the DCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x: numpy array of shape (k, d) representing initial k centers in R^d
            mu: Current smoothing parameter
            ord: norm type, example: 1, 2, np.inf
            proj_func: Function handle for projection operator
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of inner iterations
            norm_iters: List of norm differences
            cost_iters: List of cost values

            logs: Dictionary containing logs for this inner loop
        """
        k, d = x.shape
        m = a.shape[0]

        # Create a matrix of ones(k, m) * a
        s = np.ones((k, m)).dot(a)
        
        flag = True
        iter_count = 0
        cost_iters = []
        norm_iters = []
        solution = []
        
        while flag:
            iter_solution = {}
            x_old = x.copy()
            cost_old, indices = self._compute_cost(a, x, ord)
            iter_solution["indices"] = indices
            cost_iters.append(cost_old)
            
            # Compute w and z terms
            w = self._compute_w(x, a, proj_func, mu, ord)
            z = self._compute_z(x, a, proj_func, mu, ord)

            # Update x
            x = (mu * (w + z) + s) / m
            iter_solution["centers"] = x
            # Compute new cost and norm difference
            _, indices = self._compute_cost(a, x)
            # norm_diff = abs(cost_old - cost_new)
            norm_diff = np.linalg.norm(x_old - x, 'fro')
            norm_iters.append(norm_diff)
            solution.append(iter_solution)
            iter_count += 1
            
            # Check convergence
            flag = norm_diff >= self.config.parameters.inner_norm

        return x, iter_count, norm_iters, cost_iters, solution

    def _compute_w(self, x, a, proj_func, mu=None, ord=1):
        """
        Compute the W term for DCA algorithm.
        Args:
            x: numpy array of shape (k, d) representing k centers in R^d
            a: numpy array of shape (m, d) representing m data points in R^d
            mu: smoothing parameter (not used for Linf)
        Returns:
            numpy array of shape (k, d)
        """
        k, d = x.shape
        m = a.shape[0]
        w = np.zeros((k, d))
        
        x_reshaped = x[:, None, :]
        a_reshaped = a[None, :, :]

        # Tính chuẩn
        norm = np.linalg.norm(x_reshaped - a_reshaped, ord=ord, axis=2)
        norm[norm == 0] = 1e-15  # Tránh chia cho 0

        # Tìm chỉ số nhỏ nhất theo hàng của x_a_norm
        indices = np.argmin(norm, axis=0)  # Vector cột k x 1
        
        if not mu:
            for i in range(k):
                if np.any(indices == i):
                    w[i, :] = np.sum(np.sign(x[i, :] - a[indices == i, :]), axis=0)
                else:
                    w[i, :] = 0
            diff_sign = np.sign(x_reshaped - a_reshaped)
            w = np.sum(diff_sign, axis=1) - w
        else:
            # Tính toán ma trận tạm thời (temp) cho phép chiếu
            temp = (x_reshaped - a_reshaped) / mu  # Shape: k x m x d
            temp = temp.reshape(-1, d)  # Reshape thành k*m x d
            projected = proj_func(temp).reshape(k, m, d)
            # Tính toán w
            for i in range(k):
                if np.any(indices == i):
                    w[i, :] = np.sum(projected[i, indices == i, :], axis=0)
                else:
                    w[i, :] = 0
            w = np.sum(projected, axis=1) - w
        return w

    def _compute_z(self, x, a, proj_func, mu, ord=1):
        """
        Compute the Z term for DCA algorithm.
        
        Args:
            x: numpy array of shape (k, d) representing k centers in R^d
            a: numpy array of shape (m, d) representing m data points in R^d
            mu: smoothing parameter
            
        Returns:
            numpy array of shape (k, d)
        """
        k, d = x.shape
        m = a.shape[0]
        
        x_reshaped = x[:, None, :]
        a_reshaped = a[None, :, :]
        
        # Compute the temporary term
        temp = (x_reshaped - a_reshaped) / mu
        temp_reshaped = temp.reshape(-1, d)
        
        # Project onto Lx ball
        projected = proj_func(temp_reshaped).reshape(k, m, d)

        # Compute the difference
        diff = temp - projected

        # Sum over the data points
        z = np.sum(diff, axis=1)

        return z

    def _compute_cost(self, a, x, ord=1):
        return cost(a=a, x=x, ord=ord)


class ContrainedDCA(DCA):
    def _inner_loop(self, a, x, mu, proj_func, ord=1):
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
        k, d = x.shape
        m = a.shape[0]
        tau = self.config.parameters.tau
        q = self.config.parameters.q
        
        # Create a matrix of ones(k, m) * a
        s = np.ones((k, m)).dot(a)
        
        flag = True
        iter_count = 0
        norm_iters = []
        cost_iters = []
        solution = []
        
        while flag:
            x_old = x.copy()
            iter_solution = {}
            cost, indices = self._compute_cost(a, x, ord)
            iter_solution["indices"] = indices
            iter_solution["centers"] = x
            cost_iters.append(cost)
            w = self._compute_w(x, a, proj_func, mu, ord)
            z = self._compute_z(x, a, proj_func, mu, ord)
            
            u = proj_func(x)
            y = tau * u + w + z
            
            x = (mu * y + s) / (m + mu * tau * q)
            norm_diff = np.linalg.norm(x_old - x, ord=2)
            
            norm_iters.append(norm_diff)
            solution.append(iter_solution)
            iter_count += 1
            
            flag = norm_diff >= self.config.parameters.inner_norm
        
        return x, iter_count, norm_iters, cost_iters, solution
