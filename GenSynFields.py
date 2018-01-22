'''Converstion of Roger's Matlab script to python'''
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().magic('reset -sf')

# Constants
MU0 = 4 * np.pi * 1e-7
LIFTOFF = 1e-6


def calc_dipole_parameters(declination, inclination, moment_scalar):
    '''Magnetization direction and dipole strength'''
    theta = np.deg2rad(-inclination + 90)
    phi = np.deg2rad(-declination + 90)
    moment_vector = np.zeros(3)
    moment_vector[0] = moment_scalar * np.sin(theta) * np.cos(phi)
    moment_vector[1] = moment_scalar * np.sin(theta) * np.sin(phi)
    moment_vector[2] = moment_scalar * np.cos(theta)
    return moment_vector


def calc_observation_grid(x_bound, y_bound):
    '''Generate observation coordinates'''
    n_x_points = 251
    n_y_points = 251
    x_grid, y_grid = np.meshgrid(np.linspace(-x_bound, x_bound, n_x_points),
                                 np.linspace(-y_bound, y_bound, n_y_points))
    return x_grid, y_grid


def calc_point_source_field(moment_vector, z_raw, x_grid, y_grid):
    '''Compute the field of a magnetic dipole point source'''
    z_observed = z_raw * LIFTOFF
    squared_distance = (x_grid**2.0 + y_grid**2.0 + z_observed**2.0)
    aux = (moment_vector[0] * x_grid +
           moment_vector[1] * y_grid +
           moment_vector[2] * z_observed) / \
           squared_distance**(5.0 / 2.0)
    bz_dip = MU0 / (4.0 * np.pi) * \
            (3.0 * aux * z_observed - moment_vector[2] / squared_distance**(3.0 / 2.0))
    return bz_dip


def sample_sphere_angles(n_points):
    '''From: https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere'''
    declination = np.rad2deg(2*np.pi*np.random.rand(n_points))
    inclination = np.rad2deg(np.arccos(2*np.random.rand(n_points)-1)) - 90
    return declination, inclination


def gen_point_source_parameters(n_points, x_bound, y_bound):
    '''Generate diction of point source parameters'''
    point_source = dict()
    point_source['declination'], \
    point_source['inclination'] = sample_sphere_angles(n_points)
    point_source['x_source'] = np.random.uniform(-x_bound,
                                                 x_bound,
                                                 n_points)
    point_source['y_source'] = np.random.uniform(-y_bound,
                                                 y_bound,
                                                 n_points)
    point_source['moment_scalar'] = np.random.uniform(1e-14,
                                                      1e-12,
                                                      n_points)
    
    point_source['moment_vector'] = np.zeros((n_points, 3))
    for i in range(n_points):
        point_source['moment_vector'][i, :] = calc_dipole_parameters(point_source['declination'][i],
                                                                     point_source['inclination'][i],
                                                                     point_source['moment_scalar'][i])

    return point_source

def main():
    '''Generate random fields'''
    # Variables for current point source
    n_points = 100
    x_bound = 500.0e-6
    y_bound = x_bound

    point_source = gen_point_source_parameters(n_points,
                                               x_bound,
                                               y_bound)

    print point_source.keys()
    # declination = 18.0 # degrees
    # inclination = -45.0 # degrees
    z_raw = 110 # microns
    # moment_scalar = 1.0e-13 # units???

    x_grid, y_grid = calc_observation_grid(x_bound,
                                           y_bound)

    # bz_dip = calc_point_source_field(moment_vector,
    #                                  z_raw,
    #                                  x_grid,
    #                                  y_grid)

    # plt.imshow(bz_dip)
    # plt.show()


if __name__ == '__main__':
    main()
