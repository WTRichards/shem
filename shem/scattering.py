import os, sys

import numpy as np
import cupy as cp
import shem.geometry
import shem.ray
from   shem.definitions import *


# TODO: Generalise this like the source function.
def scattering_function(a, faces, surface, func):
    xp = cp.get_array_module(a, surface.vertices, faces)
    
    r_dim = a.shape[0]
    out = xp.empty_like(a)
    choice = xp.random.rand(r_dim)

    thresholds = xp.zeros(len(func)+1, dtype=xp.float32)
    for i, k in enumerate(func):
        thresholds[i+1] = func[k][1]
    thresholds = thresholds.cumsum() / thresholds.sum()

    # Choose whether each ray is reflected in a specular or diffuse manner.
    for i, k in enumerate(func):
        chosen_function = xp.logical_and(thresholds[i] < choice, choice < thresholds[i+1])
        incoming_vectors = a[chosen_function]
        faces_hit        = faces[chosen_function]
        out[chosen_function] = func[k][0](incoming_vectors, faces_hit, surface, **func[k][2])

    return out

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
        t, scattered_rays_indices, scattered_faces_indices = shem.ray.detect_collisions(r_f[:, scattered, :], surface) 
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

        a[scattered] = scattering_function(a_, scattered_faces_indices, surface, func)
        # Add a little bit to b so that the ray is scattered just above the surface.
        # This prevents collision detection weirdness.
        b[scattered] = shem.ray.parameterise(a_, b_, t) + DELTA*n_

    return r_f, n_s


"""
def calculate(r, s, conf, d, d_r, displacement, face_properties=None):
    xp = cp.get_array_module(r,d,d_r)
    
    # Surface normals
    n = s.normals
    
    # Array sizes
    d_dim     = d_r.size
    x_dim     = r.shape[1]
    y_dim     = r.shape[2]
    nparallel = r.shape[3]
    f_dim     = n.shape[0]

    # Shape of the number of rays we are tracing
    original_shape = (x_dim, y_dim, nparallel)
    original_indices = xp.arange(x_dim*y_dim*nparallel)

    # Assume all detectors point straight down.
    detector_normal_default = xp.array([0,0,-1])
    d_n = xp.tile(detector_normal_default, d_dim)

    # Initialise output tensor.
    out = xp.zeros((d_dim, x_dim, y_dim))
    
    # Rename the variables of interest to match the loop and reshape for convenience.
    secondary_rays = r.reshape(2,-1,3)
    a_ = secondary_rays[0]
    b_ = secondary_rays[1]
    scattered_rays_indices_original = original_indices
   
    # Perform N scatterings
    for i in range(conf.n+1):
        # Detect which faces the rays collide with.
        # We need to keep track of which ray is which, so return indices.
        t, scattered_rays_indices, scattered_faces_indices = shem.ray.detect_collisions(secondary_rays, s)

        # Determine the rays which did not collide with the surface.
        # If we were strictly using NumPy we could use np.delete.
        not_scattered = xp.ones(secondary_rays.shape[1], dtype=xp.bool)
        not_scattered[scattered_rays_indices] = False
        not_scattered_rays_indices_original = xp.unravel_index(scattered_rays_indices_original[not_scattered], original_shape)
        
        # Back-reference the indices to their original source ray.
        scattered_rays_indices_original = scattered_rays_indices_original[scattered_rays_indices]
    
        # Detect whether they enter a detector.
        out += shem.ray.detected(
                secondary_rays[:, not_scattered, :],
                not_scattered_rays_indices_original,
                displacement,
                original_shape,
                d, d_r, d_n)

        # Return now if there are no collisions.
        if t is None:
            return out
       
        # Calculate the secondary rays produced by scattering.
        a_ = a_[scattered_rays_indices]
        b_ = b_[scattered_rays_indices]
        n_ = n[scattered_faces_indices]
        secondary_rays = xp.stack((
            scattering_function(conf, a_, n_, scattered_faces_indices, face_properties),
            # Add a little bit to b so that the ray is scattered just above the surface.
            # This prevents collision detection weirdness.
            shem.ray.parameterise(a_, b_, t) + DELTA*n_,
        ))

       
    return out
"""
