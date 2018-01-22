'''Converstion of Roger's Matlab script to python'''
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().magic('reset -sf')

# Constants
MU0 = 4 * np.pi * 1e-7
LIFTOFF = 1e-6


def calc_dipole_parameters(dec, inc, moment_scalar):
    '''Magnetization direction and dipole strength'''
    # theta = np.radians(-dec + 90)
    # phi = np.radians(-inc + 90)
    theta = np.radians(-inc + 90)
    phi = np.radians(-dec + 90)
    

    moment_vector = dict()
    moment_vector['x'] = moment_scalar * np.sin(theta) * np.cos(phi)
    moment_vector['y'] = moment_scalar * np.sin(theta) * np.sin(phi)
    moment_vector['z'] = moment_scalar * np.cos(theta)
    return moment_vector


def calc_observation_grid():
    '''Generate observation coordinates'''
    x_bound = 500.0e-6
    y_bound = x_bound
    n_x_points = 251
    n_y_points = 251
    x_grid, y_grid = np.meshgrid(np.linspace(-x_bound, x_bound, n_x_points),
                                 np.linspace(-y_bound, y_bound, n_y_points))
    return x_grid, y_grid


def calc_point_source_field(moment_vector, z_raw, x_grid, y_grid):
    '''Compute the field of a magnetic dipole point source'''
    z_observed = z_raw * LIFTOFF
    squared_distance = (x_grid**2.0 + y_grid**2.0 + z_observed**2.0)
    aux = (moment_vector['x'] * x_grid +
           moment_vector['y'] * y_grid +
           moment_vector['z'] * z_observed) / \
           squared_distance**(5.0 / 2.0)
    bz_dip = MU0 / (4.0 * np.pi) * \
            (3.0 * aux * z_observed - moment_vector['z'] / squared_distance**(3.0 / 2.0))
    return bz_dip


def main():
    '''Generate random fields'''
    # Variables for current point source
    declination = 18.0 # degrees
    inclination = -45.0 # degrees
    z_raw = 110 # microns
    moment_scalar = 1.0e-13 # units???

    x_grid, y_grid = calc_observation_grid()

    moment_vector = calc_dipole_parameters(declination,
                                           inclination,
                                           moment_scalar)

    bz_dip = calc_point_source_field(moment_vector,
                                     z_raw,
                                     x_grid,
                                     y_grid)

    plt.imshow(bz_dip)
    plt.show()


if __name__ == '__main__':
    main()
