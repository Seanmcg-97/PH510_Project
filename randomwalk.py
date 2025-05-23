#!/bin/python3
"""
This module contains the set-up and calculation of a random walk generator with 
set boundary and charge distribution conditions, to calculate the Green's
function through Monte Carlo simulations.

MIT License

Copyright (c) 2025 Sean McGeoghegan

See LICENSE.txt for details
"""

import numpy as np
from mpi4py import MPI
from class_1 import MonteCarlo

# MPI Initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

# Initial set up
N = int(1000) # Number of iterations for Monte Carlo simulation
GRID_SIZE = 101  # The grid must be odd to give center point
STEP = 10/100  # STEP = 0.1cm
grid_tot = np.zeros([GRID_SIZE,GRID_SIZE])

# with GRID_SIZE = 101, the following arrays set the grid starting point in i, j
start_i = np.array([int(100/2), int(100/4), int(10/100), int(10/100)])
start_j = np.array([int(100/2), int(100/4), int(100/4), int(10/100)])

#---------------Task 2-------------------

def boundary_conditions(GRID_SIZE, i_zero, i_n, j_zero, j_n):
    """
    This function sets user specific conditions for 
    the grid boundaries.

    Parameters
    ----------
    GRID_SIZE : INT
        The length of the grid
    i_zero : FLOAT
        Value chosen for the zero index of i
    i_n : FLOAT
        Value chosen for the n index of i
    j_zero : FLOAT
        Value chosen for the zero index of j
    j_n : FLOAT
        Value chosen for the n index of j

    Returns
    -------
    grid : ARRAY
        A grid with boundary conditions encapsuling an empty grid

    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    # Sets the boundary conditions
    grid[0, :] = i_zero      # Top row
    grid[-1, :] = i_n        # Bottom row
    grid[:, 0] = j_zero      # Left column
    grid[:, -1] = j_n        # Right column

    return grid

# sets a 1C charge within the boundaries of the grid
uniform_grid = np.zeros_like(grid_tot)
uniform_grid[1:-1, 1:-1] = 10/((99)**2)

# forms a linear charge scale which decreases in the y direction of the grid
linear_grid = np.zeros_like(grid_tot)
charge_gradient = np.linspace(1, 0, 99)
for i in range(1, len(linear_grid)-1):
    linear_grid[i, 1:-1] = charge_gradient[i-1] * STEP**2

# placed a point charge at centre of the grid to display an exponential decay
exp_grid = np.zeros_like(grid_tot)
for i in range(1, 100):
    for j in range(1, 100):
        r = np.sqrt(((i - 50)*STEP)**2 + ((j - 50)*STEP)**2)
        exp_grid[:, :] = np.exp(-2000*np.abs(r))

# BOUNDARIES for:
# +1V all around
uniform_boundary = boundary_conditions(GRID_SIZE, 1, 1, 1, 1)

# Top/bottom = 1V, left/right = -1V
mixed_boundary = boundary_conditions(GRID_SIZE, 1, 1, -1, -1)

# Top/left = 2V, bottom = 0V, right = -4V 
complex_boundary = boundary_conditions(GRID_SIZE, 2, 0, 2, -4)

def randomwalkgen(i, j, grid):
    """
    In this function, each of the walkers for a chosen starting
    point (i, j) within a grid, will be passed through a series 
    of for loops. Each walkers path will be recorded and stored,
    to find the average probable path from all the walkers.

    Parameters
    ----------
    i : INT
        Initial starting point in x-axis
    j : INT
        Initial starting point in y-axis
    grid : ARRAY
        User defined 2 dimensional grid

    Returns
    -------
    final_grid : ARRAY
        A final array for the probable path of the walkers
    by dividing final_grid by the number of walkers

    """
    prob_grid = np.zeros_like(grid) # Empty array for storing the probablity path grid
    path = np.array([[0,1], [0,-1], [-1, 0], [1, 0]])  # Possible direction (up, down, left, right)
    n_walkers = 100 # number of walkers
    

    # This initiates each walker to begin their walk individually,
    # until they hit a user defined boundary at 0 or x/y.
    # Each walker takes one step before entering while loop.
    for n in range (0, n_walkers):
        walker_grid = np.zeros_like(prob_grid) # forms an empty grid of zeros for each walke
        start = np.array([i, j])  # setting the initial coordinates of the walk

        walk = path[np.random.randint(0, 4)] # Selects a random direction from path and sets it to walk
        start += walk # add the randomly selected direction in walk to the previous coordinates of start
        steps = 1 # sets the step taken by this walker

        # Sets a loop that while the walkers position is not
        # at a boundary, then it takes another step, until it
        # reaches a boundary
        while 0 < start[0] < len(grid)-1 and 0 < start[1] < len(grid[0])-1:
            walker_grid[start[0], start[1]] += 1  # adding instance of being at site p,q
            walk = path[np.random.randint(0, 4)]
            start += walk
            steps += 1

        # Takes the completed walk and adds it to the walkers grid
        walker_grid[start[0], start[1]] = 1

        # divide the walkers path in the inner grid
        # by steps to normalise the probability distribution
        walker_grid[1:-1, 1:-1] = walker_grid[1:-1, 1:-1]/steps 
        prob_grid += walker_grid # adds walker iteration to the probable grid array

    final_grid = prob_grid/n_walkers # final grid is found by dividing the total paths by the total walkers

    return final_grid

# Variables for each set of coordinates to pass through randomwalkgen 
# function in Monte Carlo class
variable_1 = (start_i[0], start_j[0], grid_tot)
variable_2 = (start_i[1], start_j[1], grid_tot)
variable_3 = (start_i[2], start_j[2], grid_tot)
variable_4 = (start_i[3], start_j[3], grid_tot)

# Monte Carlo sim for (5.0, 5.0)cm
mc_1 = MonteCarlo([0], [1], N, randomwalkgen, *variable_1)
results_1 = mc_1.integral()
integral_1 = results_1[0]
error_1 = results_1[2]

# Monte Carlo sim for (5.0, 5.0)cm
mc_2 = MonteCarlo([0], [1], N, randomwalkgen, *variable_2)
results_2 = mc_2.integral()
integral_2 = results_2[0]
error_2 = results_2[2]

# Monte Carlo sim for (0.1, 2.5)cm
mc_3 = MonteCarlo([0], [1], N, randomwalkgen, *variable_3)
results_3 = mc_3.integral()
integral_3 = results_3[0]
error_3 = results_3[2]

# Monte Carlo sim for (0.1, 0.1)cm
mc_4 = MonteCarlo([0], [1], N, randomwalkgen, *variable_4)
results_4 = mc_4.integral()
integral_4 = results_4[0]
error_4 = results_4[2]

if rank == 0:
    print("For (5.0, 5.0)cm")
    print(f"Error for (5.0, 5.0)cm = {error_1}")
    print("With a uniform grid")
    print(f"The potential with a uniform boundary = {np.sum(integral_1*uniform_boundary)+np.sum((STEP**2)*integral_1*uniform_grid)}":4f)
    print(f"The potential with a mixed boundary = {np.sum(integral_1*mixed_boundary)+np.sum((STEP**2)*integral_1*uniform_grid)}":4f)
    print(f"The potential with an complex boundary = {np.sum(integral_1*complex_boundary)+np.sum((STEP**2)*integral_1*uniform_grid)}":4f)
    print("With a linear grid")
    print(f"The potential with a uniform charge distribution = {np.sum(integral_1*uniform_boundary)+np.sum((STEP**2)*integral_1*linear_grid)}":4f)
    print(f"The potential with a linear charge distribution = {np.sum(integral_1*mixed_boundary)+np.sum((STEP**2)*integral_1*linear_grid)}":4f)
    print(f"The potential with an exponential decay distribution = {np.sum(integral_1*complex_boundary)+np.sum((STEP**2)*integral_1*linear_grid)}":4f)
    print("With a exponentially decaying grid")
    print(f"The potential with a uniform charge distribution = {np.sum(integral_1*uniform_boundary)+np.sum((STEP**2)*integral_1*exp_grid)}":4f)
    print(f"The potential with a linear charge distribution = {np.sum(integral_1*mixed_boundary)+np.sum((STEP**2)*integral_1*exp_grid)}":4f)
    print(f"The potential with an exponential decay distribution = {np.sum(integral_1*complex_boundary)+np.sum((STEP**2)*integral_1*exp_grid)}":4f)
    print()
    print("For (2.5, 2.5)cm")
    print(f"Error for (2.5, 2.5)cm = {error_2}")
    print("With a uniform grid")
    print(f"The potential with a uniform boundary = {np.sum(integral_2*uniform_boundary)+np.sum((STEP**2)*integral_2*uniform_grid)}":4f)
    print(f"The potential with a mixed boundary = {np.sum(integral_2*mixed_boundary)+np.sum((STEP**2)*integral_2*uniform_grid)}":4f)
    print(f"The potential with an complex boundary = {np.sum(integral_2*complex_boundary)+np.sum((STEP**2)*integral_2*uniform_grid)}":4f)
    print("With a linear grid")
    print(f"The potential with a uniform charge distribution = {np.sum(integral_2*uniform_boundary)+np.sum((STEP**2)*integral_2*linear_grid)}":4f)
    print(f"The potential with a linear charge distribution = {np.sum(integral_2*mixed_boundary)+np.sum((STEP**2)*integral_2*linear_grid)}":4f)
    print(f"The potential with an exponential decay distribution = {np.sum(integral_2*complex_boundary)+np.sum((STEP**2)*integral_2*linear_grid)}":4f)
    print("With a exponentially decaying grid")
    print(f"The potential with a uniform charge distribution = {np.sum(integral_2*uniform_boundary)+np.sum((STEP**2)*integral_2*exp_grid)}":4f)
    print(f"The potential with a linear charge distribution = {np.sum(integral_2*mixed_boundary)+np.sum((STEP**2)*integral_2*exp_grid)}":4f)
    print(f"The potential with an exponential decay distribution = {np.sum(integral_2*complex_boundary)+np.sum((STEP**2)*integral_2*exp_grid)}":4f)
    print()
    print("For (0.1, 2.5)cm")
    print(f"Error for (0.1, 2.5)cm = {error_3}")
    print("With a uniform grid")
    print(f"The potential with a uniform boundary = {np.sum(integral_3*uniform_boundary)+np.sum((STEP**2)*integral_3*uniform_grid)}":4f)
    print(f"The potential with a mixed boundary = {np.sum(integral_3*mixed_boundary)+np.sum((STEP**2)*integral_3*uniform_grid)}":4f)
    print(f"The potential with an complex boundary = {np.sum(integral_3*complex_boundary)+np.sum((STEP**2)*integral_3*uniform_grid)}":4f)
    print("With a linear grid")
    print(f"The potential with a uniform charge distribution = {np.sum(integral_3*uniform_boundary)+np.sum((STEP**2)*integral_3*linear_grid)}":4f)
    print(f"The potential with a linear charge distribution = {np.sum(integral_3*mixed_boundary)+np.sum((STEP**2)*integral_3*linear_grid)}":4f)
    print(f"The potential with an exponential decay distribution = {np.sum(integral_3*complex_boundary)+np.sum((STEP**2)*integral_3*linear_grid)}":4f)
    print("With a exponentially decaying grid")
    print(f"The potential with a uniform charge distribution = {np.sum(integral_3*uniform_boundary)+np.sum((STEP**2)*integral_3*exp_grid)}":4f)
    print(f"The potential with a linear charge distribution = {np.sum(integral_3*mixed_boundary)+np.sum((STEP**2)*integral_3*exp_grid)}":4f)
    print(f"The potential with an exponential decay distribution = {np.sum(integral_3*complex_boundary)+np.sum((STEP**2)*integral_3*exp_grid)}":4f)
    print()
    print("For (0.1, 0.1)cm")
    print(f"Error for (0.1, 0.1)cm = {error_4}")
    print("With a uniform grid")
    print(f"The potential with a uniform boundary = {np.sum(integral_4*uniform_boundary)+np.sum((STEP**2)*integral_4*uniform_grid)}":4f)
    print(f"The potential with a mixed boundary = {np.sum(integral_4*mixed_boundary)+np.sum((STEP**2)*integral_4*uniform_grid)}":4f)
    print(f"The potential with an complex boundary = {np.sum(integral_4*complex_boundary)+np.sum((STEP**2)*integral_4*uniform_grid)}":4f)
    print("With a linear grid")
    print(f"The potential with a uniform charge distribution = {np.sum(integral_4*uniform_boundary)+np.sum((STEP**2)*integral_4*linear_grid)}":4f)
    print(f"The potential with a linear charge distribution = {np.sum(integral_4*mixed_boundary)+np.sum((STEP**2)*integral_4*linear_grid)}":4f)
    print(f"The potential with an exponential decay distribution = {np.sum(integral_4*complex_boundary)+np.sum((STEP**2)*integral_4*linear_grid)}":4f)
    print("With a exponentially decaying grid")
    print(f"The potential with a uniform charge distribution = {np.sum(integral_4*uniform_boundary)+np.sum((STEP**2)*integral_4*exp_grid)}":4f)
    print(f"The potential with a linear charge distribution = {np.sum(integral_4*mixed_boundary)+np.sum((STEP**2)*integral_4*exp_grid)}":4f)
    print(f"The potential with an exponential decay distribution = {np.sum(integral_4*complex_boundary)+np.sum((STEP**2)*integral_4*exp_grid)}":4f)
    print()
    
MPI.Finalize()
