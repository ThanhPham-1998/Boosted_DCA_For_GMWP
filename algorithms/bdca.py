import numpy as np

from .base import BaseBDCA, BaseConstrainedBDCAV2
from .dca import DCA
from cost import cost, cost_pen_opt


class BDCA(BaseBDCA):
    def _inner_loop(self, a, x, mu, proj_func, ord=1, use_self_adaptive=True):
        """
        Inner loop of the BDCA algorithm.
        
        Args:
            a: numpy array of shape (m, d) representing m data points in R^d
            x: numpy array of shape (k, d) representing initial k centers in R^d
            mu: Current smoothing parameter
            proj_func: Function handle for projection operator
            ord: norm type, example: 1, 2, np.inf
            use_self_adaptive: Flag to turn on self-adaptivity for lambda in BDCA
            
        Returns:
            x: numpy array of shape (k, d) representing optimized centers
            iter_count: Number of inner iterations
            norm_iters: List of norm differences
            cost_iters: List of cost values
            lambda_iters: List of lambda values
            search_iters: List of search values
            solution: List of solutions
        """

        k, d = x.shape
        m = a.shape[0]
        tau = self.config.parameters.tau
        q = self.config.parameters.q
        
        alpha = self.config.parameters.alpha
        beta = self.config.parameters.beta
        lambda_start = self.config.parameters.lambda_start
        gamma = self.config.parameters.gamma
        lambda_history_length = self.config.parameters.lambda_history
        max_search = self.config.parameters.max_search
        lambda_min = self.config.parameters.lambda_min
        lambda_skip = self.config.parameters.lambda_min
        
        s = np.ones((k, m)).dot(a)
        flag = True
        iter_count = 0
        search_iters = []
        lambda_iters = []
        norm_iters = []
        lambda_hist = np.zeros(lambda_history_length)
        lambda_bar = np.zeros(lambda_history_length)
        lambda_val = lambda_start
        lambda_skip_iter = lambda_skip
        solution = []
        cost_iters = []

        while flag:
            x_old = x.copy()
            iter_solution = dict()
            cost_old, indices = self._compute_cost(a, x, ord)
            iter_solution['indices'] = indices
            cost_iters.append(cost_old)

            # Compute w and z terms
            w = self._compute_w(x, a, proj_func, mu, ord)
            z = self._compute_z(x, a, proj_func, mu, ord)

            # Compute x_k in DCA step
            x_k = (mu * (w + z) + s) / m

            if lambda_val < lambda_min and lambda_skip_iter == lambda_skip:
                lambda_hist = np.full(lambda_history_length, lambda_min)
                lambda_val = gamma * lambda_min
                search_iters.append(-1)
                lambda_iters.append(0)
                lambda_skip_iter = 0
                x = x_k

            elif lambda_skip_iter < lambda_skip:
                lambda_skip_iter += 1
                search_iters.append(-1)
                lambda_iters.append(0)
                x = x_k

            else:
                if use_self_adaptive:
                    lambda_hist[:-1] = lambda_hist[1:]
                    lambda_hist[-1] = lambda_val

                    if np.all(np.abs(lambda_bar - lambda_hist) < 1e-6):
                        lambda_val = gamma * lambda_hist[-1]
                    else:
                        # Use previous lambda
                        lambda_val = lambda_hist[-1]
                    
                    # Update lambda_bar history
                    lambda_hist[:-1] = lambda_hist[1:]
                    lambda_hist[-1] = lambda_val
                else:
                    lambda_val = lambda_start
                

                # Compute direction
                d_k = x_k - x_old
                
                # Update x with boosting
                x = x_k + (d_k * lambda_val)
                
                # Compute cost values
                phi_k = self._compute_cost_pen_opt(a, x_k, ord, mu, proj_func)
                cost_pen = self._compute_cost_pen_opt(a, x, ord, mu, proj_func)
                
                # Compute right-hand side of the condition
                rhs = alpha * np.linalg.norm(d_k, 'fro')**2
                d = phi_k - lambda_val**2 * rhs
                
                # Line search
                search_iter = 0
                while cost_pen > d and search_iter < max_search:
                    lambda_val = lambda_val * beta
                    x = x_k + (d_k * lambda_val)
                    cost_pen = self._compute_cost_pen_opt(a, x, mu, ord, proj_func)
                    d = phi_k - lambda_val**2 * rhs
                    search_iter += 1
                
                search_iters.append(search_iter)
                lambda_iters.append(lambda_val)
                
            # Compute norm of the difference
            norm_diff = np.linalg.norm(x_old - x, 'fro')
            norm_iters.append(norm_diff)
            iter_solution["centers"] = x
            flag = norm_diff > self.config.parameters.inner_norm
            solution.append(iter_solution)
            iter_count += 1

        return x, iter_count, norm_iters, cost_iters, search_iters, lambda_iters, solution

    def _compute_cost(self, a, x, ord=1):
        return cost(a, x, ord)

    def _compute_cost_pen_opt(self, a, x, ord, mu, proj_func):
        return cost_pen_opt(a, x, mu, ord, proj_func)

    def _compute_w(self, x, a, proj_func, mu=None, ord=1):
        return DCA._compute_w(self, x, a, proj_func, mu, ord)
    
    def _compute_z(self, x, a, proj_func, mu=None, ord=1):
        return DCA._compute_z(self, x, a, proj_func, mu, ord)


class ConstrainedBDCA(BDCA):
    def _inner_loop(self, a, x, mu, proj_func, ord=1, use_self_adaptive=True):
        k, d = x.shape
        m = a.shape[0]
        tau = self.config.parameters.tau
        q = self.config.parameters.q
        
        alpha = self.config.parameters.alpha
        beta = self.config.parameters.beta
        lambda_start = self.config.parameters.lambda_start
        gamma = self.config.parameters.gamma
        lambda_history_length = self.config.parameters.lambda_history
        max_search = self.config.parameters.max_search
        lambda_min = self.config.parameters.lambda_min
        lambda_skip = self.config.parameters.lambda_min
        
        s = np.ones((k, m)).dot(a)
        flag = True
        iter_count = 0
        search_iters = []
        lambda_iters = []
        norm_iters = []
        lambda_hist = np.zeros(lambda_history_length)
        lambda_bar = np.zeros(lambda_history_length)
        lambda_val = lambda_start
        lambda_skip_iter = lambda_skip
        solution = []
        cost_iters = []
        
        while flag:
            x_old = x.copy()
            iter_solution = {}
            cost_old, indices =self._compute_cost(a, x, ord)
            iter_solution["indices"] = indices
            cost_iters.append(cost_old)

            w = self._compute_w(x, a, proj_func, mu, ord)
            z = self._compute_z(x, a, proj_func, mu, ord)

            u = proj_func(x)
            y = tau * u + w + z
            x_k = (mu * y + s) / (m + mu * tau * q)
            
            if lambda_val < lambda_min and lambda_skip_iter == lambda_skip:
                lambda_hist = np.full(lambda_history_length, lambda_min)
                lambda_val = gamma * lambda_min
                search_iters.append(-1)
                lambda_iters.append(0)
                lambda_skip_iter = 0
                x = x_k

            elif lambda_skip_iter < lambda_skip:
                lambda_skip_iter += 1
                search_iters.append(-1)
                lambda_iters.append(0)
                x = x_k

            else:
                if use_self_adaptive:
                    lambda_hist[:-1] = lambda_hist[1:]
                    lambda_hist[-1] = lambda_val

                    if np.all(np.abs(lambda_bar - lambda_hist) < 1e-6):
                        lambda_val = gamma * lambda_hist[-1]
                    else:
                        # Use previous lambda
                        lambda_val = lambda_hist[-1]
                    
                    # Update lambda_bar history
                    lambda_hist[:-1] = lambda_hist[1:]
                    lambda_hist[-1] = lambda_val
                else:
                    lambda_val = lambda_start
                

                # Compute direction
                d_k = x_k - x_old
                
                # Update x with boosting
                x = x_k + (d_k * lambda_val)
                
                # Compute cost values
                phi_k = self._compute_cost_pen_opt(a, x_k, ord, mu, proj_func)
                cost_pen = self._compute_cost_pen_opt(a, x, ord, mu, proj_func)
                
                # Compute right-hand side of the condition
                rhs = alpha * np.linalg.norm(d_k, 'fro') ** 2
                d = phi_k - lambda_val ** 2 * rhs

                # Line search
                search_iter = 0
                while cost_pen > d and search_iter < max_search:
                    lambda_val = lambda_val * beta
                    x = x_k + (d_k * lambda_val)
                    cost_pen = self._compute_cost_pen_opt(a, x, ord, mu, proj_func)
                    d = phi_k - lambda_val ** 2 * rhs
                    search_iter += 1
                
                search_iters.append(search_iter)
                lambda_iters.append(lambda_val)
                
            # Compute norm of the difference
            norm_diff = np.linalg.norm(x_old - x, 'fro')
            norm_iters.append(norm_diff)
            iter_solution["centers"] = x
            flag = norm_diff > self.config.parameters.inner_norm
            solution.append(iter_solution)
            iter_count += 1

        return x, iter_count, norm_iters, cost_iters, search_iters, lambda_iters, solution


class ConstrainedBDCAV2(BaseConstrainedBDCAV2):
    def _inner_loop(self, a, x, mu, proj_func, ord, use_self_adaptive):
        return ConstrainedBDCA._inner_loop(self, a, x, mu, proj_func, ord, use_self_adaptive)

    def _compute_cost(self, a, x, ord=1):
        return cost(a, x, ord)

    def _compute_cost_pen_opt(self, a, x, ord, mu, proj_func):
        return cost_pen_opt(a, x, mu, ord, proj_func)

    def _compute_w(self, x, a, proj_func, mu=None, ord=1):
        return DCA._compute_w(self, x, a, proj_func, mu, ord)
    
    def _compute_z(self, x, a, proj_func, mu=None, ord=1):
        return DCA._compute_z(self, x, a, proj_func, mu, ord)