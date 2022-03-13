import numpy as np
import cupy as cp

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
        polar[..., PHI] = xp.radians(polar[..., PHI])
    
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

    polar[..., R] = xp.linalg.norm(cart, axis=-1)
    polar[..., THETA] = xp.arctan2(cart[..., Y], cart[..., X])
    polar[..., PHI] = xp.arccos(cart[..., Z] / polar[..., R])

    if not radians:
        polar[..., THETA] = xp.degrees(polar[..., THETA])
        polar[..., PHI] = xp.degrees(polar[..., PHI])

    return polar

# Generate a set of unit vectors based on a list of angular coordinates.
def unit_vector(arr_polar):
    xp = cp.get_array_module(arr_polar)
    if arr_polar.shape[-1] != 2:
        return None
    return polar2cart(xp.array([
        xp.ones(arr_polar.shape[:-1]),
        arr_polar[..., 0],
        arr_polar[..., 1],
]).T)

# Rotate a set of position vectors to be aligned with a vector n.
def rotate_vector(arr, n):
    n_polar   = cart2polar(n)
    arr_polar = cart2polar(arr)
    arr_polar[..., THETA] += n_polar[..., THETA]
    arr_polar[..., PHI]   += n_polar[..., PHI]
    return polar2cart(arr_polar)

def vector_dot(a, b):
    return (a*b).sum(-1)

def vector_normalise(a, axis=-1):
    xp = cp.get_array_module(a)
    return a / xp.expand_dims(xp.linalg.norm(a, axis=axis), axis=axis)

def vector_norm(a, axis=-1):
    xp = cp.get_array_module(a)
    return xp.linalg.norm(a, axis=axis)

def vector_cross(a, b):
    xp = cp.get_array_module(a, b)
    return xp.cross(a, b, axis=-1)
