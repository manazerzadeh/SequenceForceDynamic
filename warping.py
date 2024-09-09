import numpy as np
from scipy.optimize import minimize, dual_annealing, differential_evolution, basinhopping
import matplotlib.pyplot as plt
from pyswarm import pso
import warnings
import numba

class LinearWarping:
    def __init__(self, force_scale_flag = False):
        """
        Initialize the LinearWarping class
        
        """
        self.X = None
        self.Y = None

        self.time_points = None
        self.num_fingers = None

        self.force_scale_flag = force_scale_flag


    @numba.jit(nopython=True)
    def optimized_X_warping(X, a, b, time_points, num_fingers):
    
        time_indices = np.arange(time_points)
        warped_time_indices = a * np.arange(time_points) + b
        # warped_time_indices = np.clip(warped_time_indices, 0, time_points - 1)

        warped_X = np.empty_like(X)
        for i in range(num_fingers):
            warped_X[:, i] = np.interp(warped_time_indices, time_indices, X[:, i])
            # Manually handle out-of-bounds values
            warped_X[warped_time_indices < 0, i] = 0.0
            warped_X[warped_time_indices >= time_points, i] = 0.0


        return warped_X
    

    @numba.jit(nopython=True)
    # def optimized_X_warping_force_flag(X, a, b, S, D, time_points, num_fingers):
    def optimized_X_warping_force_flag(X, a, b, time_points, num_fingers):
    
        time_indices = np.arange(time_points)
        warped_time_indices = a * np.arange(time_points) + b
        # warped_time_indices = np.clip(warped_time_indices, 0, time_points - 1)

        warped_X = np.empty_like(X)
        for i in range(num_fingers):
            warped_X[:, i] = np.interp(warped_time_indices, time_indices, X[:, i])
            # warped_X[:,i] = S[i] * warped_X[:,i] + D[i]
            warped_X[warped_time_indices < 0, i] = 0.0
            warped_X[warped_time_indices >= time_points, i] = 0.0

        return warped_X
    

    def warp(self, X, Y):
        """
        Perform Linear Time warping between X and Y to minimzie the difference.

        :param X: numpy array of shape (time_points, num_fingers)
        :param Y: numpy array of shape (time_points, num_fingers)

        :return: optimization result and the warping parameters
        """
        self.X = X
        self.Y = Y


        assert X.shape == Y.shape, "X and Y should have the same shape"
        self.time_points = X.shape[0]
        self.num_fingers = X.shape[1]
        
        def objective(params):

            if self.force_scale_flag:
                # a, b, S, D = params[0], params[1], params[2:self.num_fingers+2], params[self.num_fingers+2:]
                a, b = params
                # warped_X = LinearWarping.optimized_X_warping_force_flag(self.X, a, b, S, D, self.time_points, self.num_fingers)
                warped_X = LinearWarping.optimized_X_warping_force_flag(self.X, a, b, self.time_points, self.num_fingers)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    result = sum([np.corrcoef(warped_X[:,i], Y[:,i])[0, 1] for i in range(self.num_fingers)])
                    if np.isnan(result):
                        return -1
                    # result = np.linalg.norm(warped_X.flatten() - Y.flatten())
                return -1 * result

            else:
                a, b = params
                warped_X = LinearWarping.optimized_X_warping(self.X, a, b, self.time_points, self.num_fingers)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    result = np.corrcoef(warped_X.flatten(), Y.flatten())[0, 1]
                if np.isnan(result):
                    return -1
                # print(result)
                return -1 * result
                    

        # initial_guess = [1.0, 0.0]

        if self.force_scale_flag:
            # S_bound = [(0.5, 1.5) for i in range(self.num_fingers)]
            # D_bound = [(-1.5, 1.5) for i in range(self.num_fingers)]
            # bounds = [(0, 3), (-self.time_points, self.time_points)] + S_bound + D_bound
            bounds = [(0, 3), (-self.time_points, self.time_points)]


        else:

            bounds = [(0, 3), (-self.time_points, self.time_points)]
        # lb = (0, -self.time_points)
        # ub = (10, self.time_points)

        initial_guess = [1.0, 0.0]

        # result = dual_annealing(objective, bounds=bounds, maxiter = 100)
        # result = basinhopping(objective, x0=[1.0, 0.0], niter=100)
        # result = differential_evolution(objective, bounds=bounds)
        result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=bounds)
        # result = pso(objective, lb, ub)

        # if not result.success:
        #     raise Exception("Optimization failed")

        return result, result.x


    def get_warped_X(self, params):
        """
        Get the warped X using the warping parameters.

        :param params: warping parameters
        :return: warped X
        """
        if self.force_scale_flag:
            a, b, S, D = params[0], params[1], params[2:self.num_fingers+2], params[self.num_fingers+2:]

        else:
            a, b = params
        warped_time_indices = a * np.arange(self.time_points) + b
        # warped_time_indices = np.clip(warped_time_indices, 0, self.time_points - 1)
        warped_X = np.array([np.interp(warped_time_indices, np.arange(self.time_points), self.X[:, i]) for i in range(self.num_fingers)]).T
        for i in range(self.num_fingers):
            # if self.force_scale_flag:
                # warped_X[:,i] = S[i] * warped_X[:,i] + D[i]
            warped_X[warped_time_indices < 0, i] = 0.0
            warped_X[warped_time_indices >= self.time_points, i] = 0.0


        return warped_X


