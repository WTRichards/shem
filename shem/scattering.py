import os, sys

import numpy as np
import cupy as cp
import shem.geometry
import shem.ray
from   shem.definitions import *

########################
# Scattering Functions #
########################

def perfect_specular(a, f, s):
    '''
    A scattering function which describes perfect specular scattering
    '''
    xp = cp.get_array_module(a, f, s.vertices)
    n = s.normals[f]
    
    out = shem.ray.reflect(a, n)

    # The incident and reflected rays should make the same angle to the normal.
    assert xp.isclose(shem.geometry.vector_dot(out, n), -shem.geometry.vector_dot(a, n)).all()

    return out

def perfect_diffuse(a, f, s):
    '''
    A scattering function which describes perfect diffuse scattering
    '''
    xp = cp.get_array_module(a, f, s.vertices)

    z = xp.array([0,0,1], dtype=xp.float32)
    n = s.normals[f]
    
    # Generate random directions
    directions = xp.random.rand(n.shape[0], 2)
    directions[:, 0] *= 2*xp.pi
    directions[:, 1] *= xp.pi
    
    # Convert the random directions into vectors.
    out = shem.geometry.unit_vector(directions)
    
    # If the ray is going into the surface, flip it so it goes out of it.
    out *= xp.expand_dims(xp.sign(shem.geometry.vector_dot(out, n)), -1)

    # The rays point out of the surface.
    assert (shem.geometry.vector_dot(out, n) > 0).all()
    
    return out

def calc_scattering_function(a, faces, surface, func):
    '''
    Calculate the result of each ray experiencing a random scattering function.
    '''
    xp = cp.get_array_module(a, surface.vertices, faces)

    r_dim = a.shape[0]
    out = xp.empty_like(a)
    choice = xp.random.rand(r_dim)

    thresholds = xp.zeros(len(func)+1, dtype=xp.float32)
    for i, k in enumerate(func):
        thresholds[i+1] = func[k]["strength"]
    thresholds = thresholds.cumsum() / thresholds.sum()

    # Choose whether each ray is reflected in a specular or diffuse manner.
    for i, k in enumerate(func):
        chosen_function = xp.logical_and(thresholds[i] < choice, choice < thresholds[i+1])
        incoming_vectors = a[chosen_function]
        faces_hit        = faces[chosen_function]
        out[chosen_function] = func[k]["function"](incoming_vectors, faces_hit, surface, **func[k]["kwargs"])

    return out

######################
# Surface Scattering #
######################

def scatter(r, surface, func, n_scat=255):
    xp = cp.get_array_module(r, surface.vertices)

    # Output rays
    r_f = r.copy()
    a = r_f[0]
    b = r_f[1]

    # Array sizes
    r_dim = r.shape[1]

    # Number of scattering events each ray undergoes.
    n_s = xp.zeros(r_dim, dtype=xp.int)
    
    # Attempt to scatter all rays once.
    scattered = xp.ones(r_dim, dtype=xp.bool)
    scattered_indices = xp.arange(r_dim)
    a_ = a
    b_ = b
    i = 0
    for i in range(n_scat):
        # Detect which faces the rays collide with.
        # We need to keep track of which ray is which, so return indices.
        t, scattered_rays_indices, scattered_faces_indices = shem.detection.detect_surface_collisions(r_f[:, scattered, :], surface)

        # Return now if there are no collisions.
        if t is None:
            break

        # Assume none of the rays collide with the surface, then record the ones that do.
        scattered_indices = scattered_indices[scattered_rays_indices]
        not_scattered = xp.ones(r_dim, dtype=xp.bool)
        not_scattered[scattered_indices] = False

        # Track how many times each ray has been scattered.
        scattered = xp.logical_not(not_scattered)
        n_s[scattered] +=1
        
        # Calculate the secondary rays produced by scattering.
        a_ =               a[scattered]
        b_ =               b[scattered]
        n_ = surface.normals[scattered_faces_indices]

        # Add a little bit to b so that the ray is scattered just above the surface.
        # This prevents collision detection weirdness.
        b[scattered] = shem.ray.parameterise(a_, b_, t) + DELTA*n_        
        a[scattered] = calc_scattering_function(a_, scattered_faces_indices, surface, func)

    return r_f, n_s

