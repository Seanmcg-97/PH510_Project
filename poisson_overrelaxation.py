#!/bin/python3
"""
This module contains the function and calculations
for the Poisson over-relaxation method used in 
task 1.

MIT License

Copyright (c) 2025 Sean McGeoghegan

See LICEnumSE.txt for details

"""
"""
Created on Wed May 14 03:53:15 2025

@author: Sean McGeoghegan
"""

import numpy as np

def poisson(n, h, num, bc, charge, init_coords):
    """
    This function finds the potential at intial starting points
    with set boundary and charge conditions using the 
    successive over-relaxation method 
    Parameters
    ----------
    n : INT
        Number of grid points
    h : FLOAT
        Step size between grid points
    num : INT
        Number of iterations
    bc : STRING
        Boundary conditions
    charge : STRING
        Charge conditions
    init_coords : ARRAY
        Initial coordinates of walker
    Returns
    -------
    xx : ARRAY
        An array of the x coordinates after over-
        relaxation method
    yy : ARRAY
        An array of the y coordinates after over-
        relaxation method
    p : ARRAY
        An array holding the potentials at init_coords

    """
    lx, ly = 10, 10  # Define square grid of side length 10 cm
    nx, ny = n, n # Define the grid points
    x = np.linspace(0, lx, nx) # length of grid in x axis
    y = np.linspace(0, ly, ny) # length of grid in y axis
    dx, dy = h, h # step size
    xx, yy = np.meshgrid(x, y) # setting the grid

    # Empty arrays for the potentials (p) and 
    # the charge within the grid (s)
    p = np.zeros((ny, nx))
    s = np.zeros((ny, nx))

    # Apply different charge distributions
    if charge == "uniform":
        s[:, :] = 10 / (nx * ny)  # 10 C spread uniformly

    elif charge == "gradient":
        charge_gradient = np.linspace(1, 0, ny)[:, np.newaxis]
        s[:, :] = charge_gradient

    elif charge == "exponential":
        center_x, center_y = lx / 2, ly / 2
        r = np.sqrt(((xx - center_x) * h)**2 + ((yy - center_y) * h)**2)
        s[:, :] = np.exp(-2000 * np.abs(r))


    w = 2 / (1 + np.sin(np.pi / num)) # Optimal parameter for OR
    tol = 1e-3 # Tolerance
    error = 1e10 # Error
    iteration = 0 # current number of iterations
    max_iteration = num # maximum number of iterations

    while error > tol and iteration < max_iteration:
        # Copy the previous iteration of while loop
        p_k = p.copy()

        # Apply boundary conditions
        p[-1, :] = bc[0]  # Top wall
        p[0, :] = bc[1]   # Bottom wall
        p[:, 0] = bc[2]   # Left wall
        p[:, -1] = bc[3]  # Right wall

        # Relaxation process
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Successive over-relaxation method
                p[j, i] = (
                    w * 1 / (2 * (dx**2 + dy**2)) * (
                        (dx**2) * (dy**2) * s[j, i] +
                        (dy**2) * (p[j, i+1] + p[j, i-1]) +
                        (dx**2) * (p[j+1, i] + p[j-1, i])
                    ) + (1 - w) * p_k[j, i]
                )

        diff = p - p_k
        error = np.linalg.norm(diff, 2)
        iteration += 1

    if iteration == max_iteration:
        print("Solution did not converge:", iteration, "Iterations")
        print("Error =", error)
    else:
        print("Solution has converged in:", iteration, "Iterations")

    # Empty array for potentials
    p = []

    for in_x, in_y in init_coords:
        idx_x = np.argmin(np.abs(x - i))  # Find nearest grid index for x
        idx_y = np.argmin(np.abs(y - j))  # Find nearest grid index for y
        
        numerical_potential = p[idx_y, idx_x]
        p.append((in_x, in_y, f"{numerical_potential:.4g}"))

    return xx, yy, p

init_coords = [(5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]

bc_1 = np.array([1, 1, 1, 1])
bc_2 = np.array([1, 1, -1, -1])
bc_3 = np.array([2, 0, 2, -4])

xx_1a, yy_1a, p_1a = poisson(100, 0.1, 5000, bc=[1, 1, 1, 1], charge="uniform", init_coords=init_coords)
xx_1b, yy_1b, p_1b = poisson(100, 0.1, 5000, bc=[1, 1, -1, -1], charge="uniform", init_coords=init_coords)
xx_1c, yy_1c, p_1c = poisson(100, 0.1, 5000, bc=[2, 0, 2, -4], charge="uniform", init_coords=init_coords)

print(f"Potentials = {p_1a} for all bc at 1V in uniform")
print(f"Potentials = {p_1b} for T/B = 1V, L/R = -1V in uniform")
print(f"Potentials = {p_1c} for T = 2V, B = 0V L = 2V & R = -4V in uniform")

xx_2a, yy_2a, p_2a = poisson(100, 0.1, 5000, bc=[1, 1, 1, 1], charge="gradient", init_coords=init_coords)
xx_2b, yy_2b, p_2b = poisson(100, 0.1, 5000, bc=[1, 1, -1, -1], charge="gradient", init_coords=init_coords)
xx_2c, yy_2c,p_2c = poisson(100, 0.1, 5000, bc=[2, 0, 2, -4], charge="gradient", init_coords=init_coords)

print(f"Potentials = {p_2a} for all bc at 1V in charge grad.")
print(f"Potentials = {p_2b} for T/B = 1V, L/R = -1V in charge grad.")
print(f"Potentials = {p_2c} for T = 2V, B = 0V L = 2V & R = -4V in charge grad.")

xx_3a, yy_3a, p_3a = poisson(100, 0.1, 5000, bc=[1, 1, 1, 1], charge="exponential", init_coords=init_coords)
xx_3b, yy_3b, p_3b = poisson(100, 0.1, 5000, bc=[1, 1, -1, -1], charge="exponential", init_coords=init_coords)
xx_3c, yy_3c, p_3c = poisson(100, 0.1, 5000, bc=[2, 0, 2, -4], charge="exponential", init_coords=init_coords)

print(f"Potentials = {p_3a} for all bc at 1V in exponential grid")
print(f"Potentials = {p_3b} for T/B = 1V, L/R = -1V in exponential grid")
print(f"Potentials = {p_3c} for T = 2V, B = 0V L = 2V & R = -4V in exponential grid")
