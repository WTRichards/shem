import os, sys

import numpy as np
import cupy as cp
import shem.geometry
import shem.ray
from   shem.definitions import *

########################
# Scattering Functions #
########################

def specular_perfect(a, f, s):
    '''
    A scattering function which describes perfect specular scattering
    '''
    xp = cp.get_array_module(a, f, s.vertices)
    n = s.normals[f]
    
    out = shem.ray.reflect(a, n)

    # The incident and reflected rays should make the same angle to the normal.
    assert xp.isclose(shem.geometry.vector_dot(out, n), -shem.geometry.vector_dot(a, n)).all()

    return out

def specular_inelastic(a, f, s, mu=0.0, sigma=1.0):
    '''
    Add a normally distributed 'kick' to the specular ray in the direction of the normal.
    '''
    xp = cp.get_array_module(a, f, s.vertices)
    n = s.normals[f]
    return specular_perfect(a, f, s) + n*xp.expand_dims(xp.random.rand(n.shape[0]), -1)

def diffuse_perfect(a, f, s):
    '''
    A scattering function which describes perfect diffuse scattering
    '''
    xp = cp.get_array_module(a, f, s.vertices)
    n = s.normals[f]
    
    # Generate random directions
    directions = shem.geometry.vector_normalise(xp.random.randn(*n.shape))
    
    # If the ray is going into the surface, flip it so it goes out of it.
    directions *= xp.expand_dims(xp.sign(shem.geometry.vector_dot(directions, n)), -1)

    # The rays point out of the surface.
    assert (shem.geometry.vector_dot(directions, n) > 0).all()
    
    return directions

def diffuse_specular_superposition(a, f, s, r=0.9):
    '''
    A linear superposition of specular and diffuse with some ratio r of specular to diffuse.
    '''
    xp = cp.get_array_module(a, f, s.vertices)
    return shem.geometry.vector_normalise(r*specular_perfect(a,f,s) + (1-r)*diffuse_perfect(a,f,s))


def diffraction_2d_simple(a, f, s, sigma_envelope=1.0, sigma_peaks=0.1):
    '''
    Calculate the 2D diffraction pattern for a particular surface.
    We assume the surface properties encodes two reciprocal lattice vectors.
    When the helium atom collides with the surface, its momentum change is quantised.
    Here, we will assume the probability of scattering by exchanging a phonon momentum k is normallly distributed.
    '''
    xp = cp.get_array_module(a, f, s.vertices)

    r_dim = a.shape[0]

    # Get the normals
    n = s.normals[f]
    
    # Get properties
    b1_x = s.get_property("b1_x")
    b1_y = s.get_property("b1_y")
    b1_z = s.get_property("b1_z")
    b2_x = s.get_property("b2_x")
    b2_y = s.get_property("b2_y")
    b2_z = s.get_property("b2_z")
    
    # Convert to arrays
    b1 = xp.array([b1_x, b1_y, b1_z]).T # This works if the properties are scalars, too.
    b2 = xp.array([b2_x, b2_y, b2_z]).T # This works if the properties are scalars, too.
    
    """
    # Guess the final momentum change
    envelope_guess = sigma_envelope * xp.random.randn(a.shape)

    # Decompose in the basis of the two lattice vectors and the normal
    M = xp.array([b1, b2, n]).T # Basis vector matrix
    # Components in this vector space
    L, M, _ = xp.linalg.solve(M, envelope_guess)

    # Quantise the 
    """

    # Sample the distribution across points up to max_order (in a square)
    max_order = 6

    # Determine the most likely momentum change using a Boltzman factor, scaled by sigma_envelope.
    momentum = lambda L, M: xp.expand_dims(L, -1)*b1 + xp.expand_dims(M, -1)*b2
    boltzmann_factor = lambda L, M: xp.exp( - ( shem.geometry.vector_norm(momentum(L, M)) / sigma_envelope )**2 / 4 )
    
    # Get the reciprocal lattice constants as integers
    L, M = shem.probability.pdist_sample(boltzmann_factor, n=r_dim, bounds=np.array([[-max_order, -max_order],[max_order, max_order]]), n_points=np.array([2*max_order + 1, 2*max_order + 1]), use_gpu=(xp is cp))

    # Add some uncertainty to the peaks
    L += sigma_peaks*xp.random.randn(*L.shape)
    M += sigma_peaks*xp.random.randn(*M.shape)

    # Add the momentum to the specular vector and a small random vector scaled by sigma_peaks, then normalise
    return shem.geometry.vector_normalise( specular_perfect(a, f, s) + momentum(L, M) )


def calc_scattering_function(a, faces, surface, func):
    '''
    Calculate the result of each ray experiencing a random scattering function.
    '''
    xp = cp.get_array_module(a, surface.vertices, faces)

    r_dim = a.shape[0]
    out = xp.zeros_like(a)
    choice = xp.random.rand(r_dim)

    # Calculate the final directions by __picking__ from each scattering function.
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
    a = r[0]
    b = r[1]

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
        t, scattered_rays, scattered_faces_indices = shem.detection.detect_surface_collisions(r[:, scattered, :], surface)

        # Return now if there are no collisions.
        if t is None:
            break

        # Assume none of the rays collide with the surface, then record the ones that do.
        scattered_indices = scattered_indices[scattered_rays]
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

    return r, n_s

