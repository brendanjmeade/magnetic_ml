'''
Generate a catalog of syntetic images of mulitple
monolpoles. Based on Roger Fu's Matlab script
'''

import pickle
import sys
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import cm
import matplotlib.pyplot as plt

# Constants
MU0 = 4 * np.pi * 1e-7
LIFTOFF = 5e-6 # Roger sez this should be: 5e-6
LEVELS = np.linspace(-1e-6, 1e-6, 30)
PIXELS = 300
FONT_SIZE = 14


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
    n_x_points = PIXELS
    n_y_points = PIXELS
    x_grid, y_grid = np.meshgrid(np.linspace(-2 * x_bound, 2 * x_bound, n_x_points),
                                 np.linspace(-2 * y_bound, 2 * y_bound, n_y_points))
    return x_grid, y_grid


def calc_point_source_field(moment_vector, x_source, y_source, z_raw, x_grid, y_grid):
    '''Compute the field of a magnetic dipole point source'''
    # z_observed = z_raw * LIFTOFF
    z_observed = 30e-6 * np.random.uniform(1) + LIFTOFF
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

    # Rescale moment_vector by a random number
    point_source['moment_vector'][i, :] = 1e4 * np.random.rand(1) * point_source['moment_vector'][i, :] 



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


def plot_field(grid):
    '''Simple visualization'''
    min_val = np.min(grid)
    max_val = np.max(grid)
    cbar_val = np.max(np.array([np.abs(min_val), max_val]))
    scale_factor = np.round(np.log10(cbar_val))
    cbar_val = cbar_val / (10**scale_factor)

    plt.imshow(grid / 10**(scale_factor),
               interpolation='nearest',
               cmap=cm.coolwarm,
               origin='lower')
    plt.clim(-cbar_val, cbar_val)
    plt.plot(np.array([-50, 50, 50, -50, -50])*PIXELS/200 + PIXELS/2,
             np.array([-50, -50, 50, 50, -50])*PIXELS/200 + PIXELS/2,
             '--k',
             linewidth=1.0)
    plt.xticks([0, PIXELS-1], ['-100', '100'])
    plt.yticks([0, PIXELS-1], ['-100', '100'])
    plt.xlabel('$x \; \mathrm{(microns)}$', fontsize=FONT_SIZE)
    plt.ylabel('$y \; \mathrm{(microns)}$', fontsize=FONT_SIZE)
    
    matplotlib.rc('xtick', labelsize=FONT_SIZE) 
    matplotlib.rc('ytick', labelsize=FONT_SIZE) 

    cbar = plt.colorbar(ticks=[-cbar_val, 0, cbar_val])
    exponent_string = '$10^{' + str(scale_factor) + '}$' 
    cbar.ax.set_ylabel(r'field strength (units ' + r'$\times$' + ' ' + exponent_string + ')', fontsize=14, rotation=90)
    cbar.ax.tick_params(labelsize=14) 
    plt.show(block=False)


def main(file_name):
    '''Generate random fields'''
    n_fields = 3
    n_points = 2400 # Roger says 2400
    x_bound = 100.0e-6 # microns, Roger says make this 50-100 microns
    y_bound = x_bound
    z_raw = 0 # microns, -z-coordinate: make a volume 0-30 
    n_bins = 100 # Roger wants 1 degree...I say 10 degrees
    x_grid, y_grid = calc_observation_grid(x_bound, y_bound)

    frames_bzdip = np.zeros((n_fields, PIXELS, PIXELS)) # make 300x300
    frames_moment_scalar = np.zeros(n_fields)
    frames_moment_vector = np.zeros((n_fields, 3))

    for i in range(n_fields):
        sys.stdout.write('\r [ %f'%((i+1)/n_fields * 100)+'% ] ')
        sys.stdout.flush()
        point_source = gen_point_source_parameters(n_points, x_bound, y_bound)
        point_source = field_from_point_source_dict(point_source, x_grid, y_grid, z_raw)
        frames_bzdip[i, :, :] = point_source['bz_dip']
        frames_moment_vector[i, :] = np.sum(point_source['moment_vector'], 0)
        frames_moment_scalar[i] = np.linalg.norm(frames_moment_vector[i, :])

    print(' ')
    # plt.close('all')
    # plt.figure()
    # plt.hist(frames_moment_scalar, np.logspace(-15, -10, 200))
    # plt.ylabel('N')
    # plt.xlabel('moment vector magnitude')
    # plt.show(block=False)
    # import pdb; pdb.set_trace()
    
    np.savez(file_name + '.npz',
             frames_bzdip,
             frames_moment_scalar,
             frames_moment_vector)

if __name__ == '__main__':
    main(sys.argv[1])
