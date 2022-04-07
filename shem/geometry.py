import numpy as np
import cupy as cp

from math import *
from shem.definitions import *

def polar2cart(polar_, radians=True):
    xp = cp.get_array_module(polar_)
    '''
    Accept a list of polar coordinates.
    Returns a numpy array containing the corresponding Cartesian coordinates.
    '''
    # Convert between radians and degrees.
    polar = xp.array(polar_, dtype=xp.float32)
    cart = xp.empty_like(polar)

    if not radians:
        polar[..., THETA] = xp.radians(polar[..., THETA])
        polar[...,   PHI] = xp.radians(polar[...,   PHI])
    
    # Convert to Cartesians.
    cart[..., X] = polar[..., R] * xp.sin(polar[..., PHI]) * xp.cos(polar[..., THETA])
    cart[..., Y] = polar[..., R] * xp.sin(polar[..., PHI]) * xp.sin(polar[..., THETA])
    cart[..., Z] = polar[..., R] * xp.cos(polar[..., PHI])


    return cart

def cart2polar(cart_, radians=True):
    xp = cp.get_array_module(cart_)
    '''
    Accept a list of Cartesian coordinates.
    Returns a numpy array containing the corresponding polar coordinates.
    '''
    cart = xp.array(cart_, dtype=xp.float32)
    polar = xp.empty_like(cart)

    polar[...,     R] = xp.linalg.norm(cart, axis=-1)
    polar[..., THETA] = xp.arctan2(cart[..., Y], cart[..., X])
    polar[...,   PHI] = xp.arccos( cart[..., Z] / polar[..., R])

    if not radians:
        polar[..., THETA] = xp.degrees(polar[..., THETA])
        polar[...,   PHI] = xp.degrees(polar[...,   PHI])

    return polar

# Dot product of two sets of vectors along their final axis.
def vector_dot(a, b):
    return (a*b).sum(-1)

# Get the length of a set of vectors along an axis (final by default)
def vector_norm(a, axis=-1):
    xp = cp.get_array_module(a)
    return xp.linalg.norm(a, axis=axis)

# Cross product of two sets of vectors along their final axis.
def vector_cross(a, b):
    xp = cp.get_array_module(a, b)
    return xp.cross(a, b, axis=-1)

# Normalise a set of vectors along an axis (final by default)
def vector_normalise(a, axis=-1):
    xp = cp.get_array_module(a)
    return a / xp.expand_dims(xp.linalg.norm(a, axis=axis), axis=axis)

# Generate a set of unit vectors based on a list of angular coordinates.
def unit_vector(arr_polar, radians=True):
    xp = cp.get_array_module(arr_polar)
    if arr_polar.shape[-1] != 2:
        return None
    return polar2cart(xp.array([
        xp.ones(arr_polar.shape[:-1]),
        arr_polar[..., 0],
        arr_polar[..., 1],
]).T, radians=radians)


# Rotate a set of vectors into a new coordinate frame where v0 and v1 define the z axes in the initial and final frames.
# This is useful if we want to find the vectors which can emanate from a point source in a cone, either as the source or a result of scattering.
def rotate_frame(v_i, v_f, arr):
    xp = cp.get_array_module(v_i, v_f, arr)

    # If the initial and final z axes are identical, so is the resulting array.
    if xp.array_equal(v_i, v_f):
        return arr

    # Angle between the two vectors.
    theta = xp.arccos(vector_dot(v_i, v_f) / (vector_norm(v_i) * vector_norm(v_f)) ) 
    c = xp.cos(theta)
    s = xp.sin(theta)
    
    # Cross product defining an orthonormal axis
    u = vector_normalise(xp.cross(v_i, v_f))
    u_x, u_y, u_z = u[X].item(), u[Y].item(), u[Z].item()

    # Define the rotation matrix which rotates v_i onto v_f.
    # If this were written explicitly it would look horrid.
    R = c*xp.identity(3) + (1-c)*xp.outer(u, u) + s*xp.array([
        [   0, -u_z, +u_y],
        [+u_z,    0, -u_x],
        [-u_y, +u_x,    0],
    ], dtype=xp.float32)

    # Use einstein summation rather than built-in matrix multiplication for clarity.
    return xp.einsum('ij,kj->ki', R, arr)

# Get the angle between vectors.
def vector_angle(a, b, radians=True):
    xp = cp.get_array_module(a)
    theta = xp.arccos( vector_dot(a, b) / ( vector_norm(a)*vector_norm(b) ) )
    if radians:
        return theta
    else:
        return xp.degrees(theta)
