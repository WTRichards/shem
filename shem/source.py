import os, sys
import numpy as np
import cupy as cp

from shem.definitions import *
import shem.geometry

##################
# Ray Generation #
##################

def direction(a, source_location, source_angle, coordinates, coordinate_indices):
    '''
    Determine the point source direction vector. We will trace rays in a 1 degree cone and convolve with various source distributions later.
    '''
    xp = cp.get_array_module(a, source_location, coordinates, coordinate_indices)
    
    z = xp.array([0,0,1], dtype=xp.float32)

    # Start in polar coords with a conic source function
    a_polar_i = xp.array([2*xp.pi*(xp.random.rand(a.shape[0]) - 0.5), source_angle*xp.random.rand(a.shape[0])]).T

    a[:,    R] = 1.0
    a[:,THETA] = a_polar_i[:,0]
    a[:,  PHI] = a_polar_i[:,1]

    # Convert back to Cartesians and rotate so the rays are directed from the source.
    a[:] = shem.geometry.polar2cart(a)
    a[:] = shem.geometry.rotate_frame(z, -source_location, a)

    # Convert back to polars to adjust the source angle based on the angular displacement.
    a[:] = shem.geometry.cart2polar(a)
    a[:, THETA] += coordinates[2][coordinate_indices]
    a[:,   PHI] -= coordinates[3][coordinate_indices]

    # Convert back into Cartesians
    a[:] = shem.geometry.polar2cart(a)

    return a, a_polar_i

def origin(b, source_location, coordinates, coordinate_indices):
    '''
    Determine the point source origin vector.
    '''
    xp = cp.get_array_module(b, source_location, coordinates, coordinate_indices)
    # Start in polar coords
    b[:] = shem.geometry.cart2polar(source_location)

    # Apply shift in theta and phi
    b[:, THETA] += coordinates[2][coordinate_indices]
    b[:, PHI]   += coordinates[3][coordinate_indices]

    # Convert back into cartesians
    b[:] = shem.geometry.polar2cart(b)
    
    # Apply shift in x and y
    b[:, X] -= coordinates[0][coordinate_indices]
    b[:, Y] -= coordinates[1][coordinate_indices]

    return b

####################
# Source Functions #
####################

def uniform(theta, phi):
    '''
    A uniform source function. All rays are weighted equally.
    '''
    xp = cp.get_array_module(theta, phi)
    return xp.ones_like(theta)

def cone(theta, phi, phi_max=np.radians(1)):
    '''
    A uniform cone function.
    Accepts the arguments: phi_max
    '''
    xp = cp.get_array_module(theta, phi)
    return xp.array(phi < phi_max, dtype=xp.float32)

def gaussian(theta, phi, mu=0.0, sigma=1.0):
    '''
    A Gaussian (in phi) source function.
    Accepts the arguments: mu, sigma.
    '''
    xp = cp.get_array_module(theta, phi)
    return ( 1.0 / xp.sqrt(2*xp.pi*sigma**2) ) * xp.exp( - ( phi - mu )**2 / ( 2 * sigma**2 ) )


def calc_source_function(theta, phi, settings):
    xp = cp.get_array_module(theta, phi)
    signal = xp.zeros_like(theta)
    
    func = settings["source"]["function"]
    
    # Sum the signals defined by each source function multiplied by their strengths.
    for i, k in enumerate(func):
        signal += func[k]["strength"] * func[k]["function"](theta, phi, **func[k]["kwargs"])

    return signal

