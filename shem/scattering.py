import numpy as np
import cupy as cp
import shem.geometry
import shem.ray
from   shem.definitions import *

def specular(r, s, N, d, d_r, displacement):
    return function(r, s, N, d, d_r, displacement, shem.ray.reflect)

# TODO: Get a function to output a random surface vector
def diffuse(r, s, N, d, d_r, displacement):
    return function(r, s, N, d, d_r, displacement, shem.ray.diffuse)

def function(r, s, N, d, d_r, displacement, scattering_function):
    xp = cp.get_array_module(r,s,d,d_r)
    
    # Surface normals
    n = s.normals()
    
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
    for i in range(N+1):
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
        # TODO: Implement detection functionality
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
            scattering_function(a_, n_),
            # Add a little bit to b so that the ray is scattered just above the surface.
            # This prevents collision detection weirdness.
            shem.ray.parameterise(a_, b_, t) + DELTA*n_,
        ))

       

    return out
