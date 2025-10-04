import logging
import os
from datetime import datetime
from typing import List


class Parameters:
    def __init__(
        self,
        tau: float = 1,
        sig: float = 100,
        tauf: float = 1e8,
        mu: float = 1,
        delta: float = 0.5,
        muf: float = 1e-6,
        outer_norm: float = 1e-6,
        inner_norm: float = 1e-6,
        alpha: float = 0.05,
        beta: float = 0.01,
        lambda_start: float = 1,
        gamma: float = 2,
        lambda_history: float = 1,
        max_search: int = 2,
        min_search: float = 1e-3,
        lambda_min: float = 1e-3,
        lambda_skip: int = 20
    ):
        """
        Initialize the Parameters object with default values.

        Args:
            tau: float, default 1, time constant for the filter
            sig: float, default 100, time constant for the filter
            tauf: float, default 1e8, time constant for the filter
            mu: float, default 1, smoothing parameter
            delta: float, default 0.5, reduction factor for mu
            muf: float, default 1e-6, final smoothing parameter
            outer_norm: float, default 1e-6, convergence threshold for outer iterations
            inner_norm: float, default 1e-6, convergence threshold for inner iterations
            alpha: float, default 0.05, alpha parameter for BDCA
            beta: float, default 0.01, beta parameter for BDCA
            lambda_start: float, default 1, initial lambda value
            gamma: float, default 2, gamma parameter for lambda update
            lambda_history: float, default 1, length of lambda history
            max_search: int, default 2, maximum number of search iterations
            min_search: float, default 1e-3, minimum lambda value
            lambda_skip: int, default 20, number of iterations to skip lambda search
        """
        self.tau = tau
        self.sig = sig
        self.tauf = tauf
        self.mu = mu
        self.delta = delta
        self.muf = muf
        self.outer_norm = outer_norm
        self.inner_norm = inner_norm
        self.alpha = alpha
        self.beta = beta
        self.lambda_start = lambda_start
        self.gamma = gamma
        self.lambda_history = lambda_history
        self.lambda_min = lambda_min
        self.max_search = max_search
        self.min_search = min_search
        self.lambda_skip = lambda_skip


class BaseConfig:
    def __init__(
        self,
        num_running_times: int = 1,
        dimentions: List[int] = [],
        num_points: List[int] = [],
        save_to_file: bool = False,
        parameters: Parameters = None,
        save_dir: str = "logging/",
        remove_log_history: bool = True
    ):
        
        self.num_running_times = num_running_times
        self.dimentions = dimentions
        self.num_points = num_points
        self.remove_log_history = remove_log_history
        self.parameters = parameters or Parameters()
        self.save_dir = save_dir
        if save_to_file and not parameters:
            self.save_log()
    
    def save_log(self, params: dict = None, message: str = None):
        if self.remove_log_history and os.path.exists(self.save_dir):
            try:
                os.remove(self.save_dir + "/logging.log")
            except OSError as e:
                print(f"Warning: Could not delete log file {self.save_dir}: {e}")
            self.remove_log_history = False

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir + "/logging.log", mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        if not params and not message:
            logger.info("=== Default parameters ===")
            logger.info("=== Configuration Log ===")
            logger.info(f"Number of running times: {self.num_running_times}")
            logger.info(f"Dimensions: {self.dimentions}")
            logger.info(f"Number of points: {self.num_points}")
            logger.info(f"Log file saved at: {os.path.abspath(self.save_dir)}")

            logger.info("=== Parameters ===")
            for param_name, param_value in vars(self.parameters).items():
                logger.info(f"{param_name}: {param_value}")
            
        else:
            if message:
                logger.info(f"=== {message} ===")
            if params:
                for key, value in params.items():
                    logger.info(f"{key}: {value}")
        
        logger.info(f"Log created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("======================")

        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
    
    def update_params(self, parameters: dict, save_log: bool = False):
        if parameters:
            for param, value in parameters.items():
                self.parameters.__setattr__(param, value)
            if save_log:
                self.save_log(parameters, "Update Parameters")
