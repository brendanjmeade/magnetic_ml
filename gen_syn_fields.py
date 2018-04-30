'''
Generate a catalog of syntetic images of mulitple
monolpoles. Based on Roger Fu's Matlab script
'''

import pickle
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt

# Constants
MU0 = 4 * np.pi * 1e-7
LIFTOFF = 1e-6
LEVELS = np.linspace(-1e-6, 1e-6, 30)


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
    n_x_points = 64
    n_y_points = 64
    x_grid, y_grid = np.meshgrid(np.linspace(-2 * x_bound, 2 * x_bound, n_x_points),
                                 np.linspace(-2 * y_bound, 2 * y_bound, n_y_points))
    return x_grid, y_grid


def calc_point_source_field(moment_vector, x_source, y_source, z_raw, x_grid, y_grid):
    '''Compute the field of a magnetic dipole point source'''
    z_observed = z_raw * LIFTOFF
    squared_distance = (x_grid-x_source)**2.0 + \
                       (y_grid-y_source)**2.0 + \
                       z_observed**2.0

    aux = (moment_vector[0] * (x_grid - x_source) +
           moment_vector[1] * (y_grid - y_source) +
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

def field_from_point_source_dict(point_source, x_grid, y_grid, z_raw):
    '''Calculate and sum the point sources from the point_source dict'''
    bz_dip = np.zeros(x_grid.shape)
    for i in range(point_source['moment_scalar'].size):
        bz_dip += calc_point_source_field(point_source['moment_vector'][i, :],
                                          point_source['x_source'][i],
                                          point_source['y_source'][i],
                                          z_raw,
                                          x_grid,
                                          y_grid)

    point_source['bz_dip'] = bz_dip
    return point_source


def calc_feature_labels(field, n_bins):
    _, bins, _ = plt.hist(field, n_bins, facecolor='green')
    return np.digitize(field, bins)


def main():
    '''Generate random fields'''
    n_fields = 1000000
    n_points = 50
    x_bound = 500.0e-6 # microns
    y_bound = x_bound
    z_raw = 110 # microns
    n_bins = 10
    x_grid, y_grid = calc_observation_grid(x_bound, y_bound)

    frames_bzdip = np.zeros((n_fields, 64, 64))
    frames_moment_scalar_sum = np.zeros(n_fields)
    frames_moment_vector_sum = np.zeros((n_fields, 3))
    frames_moment_vector_sum_labels = np.zeros((n_fields, 3))

    for i in range(n_fields):
        print(i+1)
        point_source = gen_point_source_parameters(n_points, x_bound, y_bound)
        point_source = field_from_point_source_dict(point_source, x_grid, y_grid, z_raw)
        frames_bzdip[i, :, :] = point_source['bz_dip']
        frames_moment_scalar_sum[i] = np.sum(point_source['moment_scalar'])
        frames_moment_vector_sum[i, :] = np.sum(point_source['moment_vector'], 0)

    # Histograms and quantized labels
    frames_moment_scalar_sum_labels = calc_feature_labels(frames_moment_scalar_sum, n_bins).astype(int)
    frames_moment_vector_sum_labels[:, 0] = calc_feature_labels(frames_moment_vector_sum[:, 0], n_bins).astype(int)
    frames_moment_vector_sum_labels[:, 1] = calc_feature_labels(frames_moment_vector_sum[:, 1], n_bins).astype(int)
    frames_moment_vector_sum_labels[:, 2] = calc_feature_labels(frames_moment_vector_sum[:, 2], n_bins).astype(int)
    
    np.savez('synthetics_100000.npz', frames_bzdip, frames_moment_scalar_sum, frames_moment_vector_sum,
             frames_moment_scalar_sum_labels, frames_moment_vector_sum_labels)

if __name__ == '__main__':
    main()
