#!/bin/python3
"""

Module containing framework for Monte Carlo simulations 
and assignment tasks

MIT License

Copyright (c) 2025 Sean McGeoghegan

See LICEnSE.txt for details

"""

from numpy.random import SeedSequence, default_rng
import numpy as np
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()


class MonteCarlo:
    """
    This class is for running Monte Carlo simulations to find the integral, 
    variance and error for the area of a circle.

    """

    def __init__(self, start, end, num, f, *varis):
        """
        Class initialisation for Monte Carlo
    
        Parameters:
        start = Starting coordinate in the square
        end = Final coordinate in the square
        f = The function being passed through
        varis = Any variables required for the function in use
        values = Class initialisation for f-string print result

        """
        self.start = start   # Starting point of integral
        self.end = end   # End point of integral
        self.f = f   # Function to be passed through MC
        self.num = num   # number of iterations to be performed by MC
        self.vars = varis   # Variable(s) within an array for a function
        self.values = 0     # Used for final results

    def __str__(self):
        """
        F-string for final results

        """
        return f"(integral: {self.values[0]}, Var: {self.values[1]}, Err: {self.values[2]})"



    def integral(self):
        """
        This function is used to calculate the integral of a function, via random walks.

        """
        d = len(self.start)

        seed = SeedSequence(12345)
        nworkers_seed = seed.spawn(nworkers)
        random_gen = default_rng(nworkers_seed[rank])
        r_num = random_gen.random((self.num, d))

        grid_sum = np.zeros((101, 101))
        grid_sum_sq = np.zeros_like(grid_sum)

        for i in range(self.num):
            result = self.f(*self.vars)
            grid_sum += result
            grid_sum_sq += result ** 2

        total_grid = np.zeros_like(grid_sum, dtype=np.float64)
        total_grid_sq = np.zeros_like(grid_sum_sq, dtype=np.float64)

        comm.Allreduce(grid_sum, total_grid)
        comm.Allreduce(grid_sum_sq, total_grid_sq)

        # Monte Carlo scaling
        total_samples = self.num * nworkers
        inv_n = 1 / total_samples
        domain_volume = np.prod(np.array(self.end) - np.array(self.start))

        # Grid mean and statistics
        mean = total_grid * inv_n
        mean_sq = total_grid_sq * inv_n
        variance_grid = mean_sq - mean ** 2
        error_grid = np.sqrt(variance_grid / total_samples)

    # Final scalar integral result (sum over full grid)
        integral = domain_volume * np.sum(mean)
        variance = np.sum(variance_grid)
        error = domain_volume * np.sqrt(variance / total_samples)

        self.values = (round(integral, 5), variance, error)
        return self.values
