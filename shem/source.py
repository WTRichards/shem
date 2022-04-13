import os, sys
import numpy as np
import cupy as cp

from shem.definitions import *
import shem.geometry

##################
# Ray Generation #
##################

# Create a vector using a probability distribution described by the polar angle phi relative to some axis z
# See: https://math.stackexchange.com/questions/56784/generate-a-random-direction-within-a-cone
def polar_distribution(a, pdist, **kwargs):
    '''
    Generate a vector using some polar angle probability distribution relative to the axis a.
    You will probably want to reduce this.
    '''
    xp = cp.get_array_module(a)

    # Sample uniformly in phi (azimuthal angle)
    phi = xp.random.rand(*a.shape[:-1])*2*xp.pi

    # Sample points in theta from the distribution in one dimension less than a i.e # of vectors.
    theta = pdist(a[..., THETA], **kwargs)

    # Calculate the z coordinate from theta.
    z = xp.cos(theta)

    # Calculate the outgoing vector relative to the z axis.
    a[:] = xp.array([xp.cos(phi)*xp.sqrt(1-z*z), xp.sin(phi)*xp.sqrt(1-z*z), z]).T

    return a


def get_source_rays(rays_i, source_location, func, coordinates, coordinate_indices):
    '''
    Get the source rays for a particular source location, function and coordinates and write it to rays_i.
    Return the deviation of the source ray from a perfect source.
    '''
    xp = cp.get_array_module(rays_i, coordinates, coordinate_indices)
    z = xp.array([0,0,1], dtype=xp.float32)

    a = rays_i[0]
    b = rays_i[1]

    #############
    # DIRECTION #
    #############

    # TODO: Calculate source distribution here.
    r_dim = a.shape[0]
    choice = xp.random.rand(r_dim)

    # Calculate the source distribution by __picking__ from each function.
    thresholds = xp.zeros(len(func)+1, dtype=xp.float32)
    for i, k in enumerate(func):
        thresholds[i+1] = func[k]["strength"]
        thresholds = thresholds.cumsum() / thresholds.sum()
    
    # Calculate the source distribution using the polar ray distribution.
    for i, k in enumerate(func):
        chosen_function = xp.logical_and(thresholds[i] < choice, choice < thresholds[i+1])
        a[chosen_function] = polar_distribution(a[chosen_function], func[k]["function"],  **func[k]["kwargs"])

    
    """
    # All rays point along the axis, for now.
    a[...,     R] = 1.0
    a[..., THETA] = 0.0
    a[...,   PHI] = 0.0
    """

    # Get the initial off-axis polar components
    a_polar_i = shem.geometry.cart2polar(a)[..., 1:]
    
    # Generate the locations of the source at each coordinate index when displaced by (theta, phi).
    angled_source_locations = shem.geometry.polar2cart( shem.geometry.cart2polar(source_location) + xp.array([xp.zeros(rays_i.shape[1]), coordinates[2][coordinate_indices], coordinates[3][coordinate_indices]]).T )
    
    # Rotate the angular distributions so they lie along the axis of a perfect source.
    a[:] = shem.geometry.rotate_frame(z, -angled_source_locations, a)
    

    # Normalise the directions
    a[:] = shem.geometry.vector_normalise(a)

    ##########
    # ORIGIN #
    ##########
    
    # Add the displacements in x and y.
    #b[:] = angled_source_locations + xp.array([coordinates[0][coordinate_indices], coordinates[1][coordinate_indices], xp.zeros(rays_i.shape[1])]).T
    b[..., :] = source_location + xp.array([coordinates[0][coordinate_indices], coordinates[1][coordinate_indices], xp.zeros(rays_i.shape[1])]).T

    # Just return the initial deviation from a perfect ray.
    return a_polar_i



####################
# Source Functions #
####################

def delta(theta):
    '''
    A perfect source function.
    '''
    xp = cp.get_array_module(theta)
    return xp.zeros_like(theta)

def uniform_cone(theta, theta_max=np.radians(0.5)):
    '''
    A uniform cone function. Returns uniformly random values of theta within a cone
    Accepts the arguments: theta_max (defaults to 0.5 degrees)
    '''
    xp = cp.get_array_module(theta)

    # Minimum value of z
    z_min = xp.cos(theta_max)
    
    # Uniform choice in range [1, z_min]
    z = (1-z_min)*xp.random.rand(*theta.shape) + z_min
    
    # Need to return arccos for the wrapper function.
    return xp.arccos(z)

def gaussian(theta, mu=0.0, sigma=1.0):
    '''
    A Gaussian (in theta) source function.
    Note that this is not integrated over phi - it is the intensity at a point at an angle theta in the cone.
    Accepts the arguments: mu, sigma.
    '''
    xp = cp.get_array_module(theta)
    return sigma*xp.random.randn(*theta.shape) + mu

