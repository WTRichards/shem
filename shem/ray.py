import os, sys

import numpy as np
import cupy as cp

import shem.geometry

from shem.definitions import *

from shem.geometry import vector_dot       as dot
from shem.geometry import vector_cross     as cross
from shem.geometry import vector_norm      as norm
from shem.geometry import vector_normalise as normalise

def parameterise(a, b, t):
    xp = cp.get_array_module(a, b, t)
    return a*xp.expand_dims(t, -1) + b

def intersects_plane_time(a, b, n, c):
    return dot(n, c - b) / dot(n, a)

def reflect(a, n):
    xp = cp.get_array_module(a, n)
    return a - 2*xp.expand_dims(dot(n, a), -1)*n

def perfect_specular(a, f, s):
    xp = cp.get_array_module(a, f, s.vertices)
    n = s.normals[f]
    return reflect(a, n)

def perfect_diffuse(a, f, s):
    xp = cp.get_array_module(a, f, s.vertices)
    n = s.normals[f]
    # Generate random directions
    directions = xp.random.rand(n.shape[0], 2)
    directions[:, 0] *= 2*xp.pi
    directions[:, 1] *= xp.pi/2
    # Rotate these random directions such that they are relative to the surface defined by the normal n.
    return shem.geometry.rotate_vector(shem.geometry.unit_vector(directions), n)

def detected(r, original_index, displacement, original_shape, d, d_r, n):
    xp = cp.get_array_module(r, displacement, d, d_r, n)

    a = r[0]
    b = r[1]
    x = original_index[0]
    y = original_index[1]
    n_p = original_index[2]
    
    r_dim = r.shape[1]
    d_dim = d_r.size
    
    # Create an array of detectors analogous to the sources corresponding to each ray.
    d_   =   (d.reshape(d_dim, 1, 1, 3) + displacement)[:, x, y, :]
    n_   =   n.reshape(d_dim, 1,     3)
    d_r_ = d_r.reshape(d_dim, 1)
    a_   =   a.reshape(1,     r_dim, 3)
    b_   =   b.reshape(1,     r_dim, 3)


    t_ = intersects_plane_time(a_, b_, n_, d_)
    p_ = parameterise(a_, b_, t_)

    # The ray is detected at time greater than zero.
    time_matches = t_ > 0
    # The ray hits the front face of the detector instead of the rear - not usually a problem.
    # normal_matches = dot(n_, a_) < 0
    # The ray is within a distance r of the center of the detector.
    within_edges = norm(p_ - d_) < d_r_

    # The matrix of truth values for which all conditions above are satisfied.
    detection_matrix = within_edges * time_matches # * normal_matches

    # Define the shape of the output array
    out = xp.zeros((d_dim, *original_shape), dtype=xp.bool)
    # Set the indices where an applicable ray has been traced to the number of rays that are detected.
    out[:, x, y, n_p] = detection_matrix

    return out.sum(-1)


def detect_collisions(r, s):
    xp = cp.get_array_module(r)

    a = r[0]
    b = r[1]
    n = s.normals
    e = s.edges
    v = s.triangles
    c = s.centroids

    r_dim = r.shape[1]
    f_dim = s.faces.shape[0]

    # Reshape arrays for minimally painful broadcasting.
    n_ = n.reshape(1,     f_dim, 1, 3)
    e_ = e.reshape(1,     f_dim, 3, 3)
    v_ = v.reshape(1,     f_dim, 3, 3)
    c_ = c.reshape(1,     f_dim, 1, 3)
    a_ = a.reshape(r_dim, 1,     1, 3)
    b_ = b.reshape(r_dim, 1,     1, 3)

    # Calculate the time each ray intersects with each plane and expand for broadcasting.
    t_ = intersects_plane_time(a_, b_, n_, c_)

    # Point each ray intersects with each plane.
    p_ = parameterise(a_, b_, t_)
    # The rays intersect at time greater than zero.
    time_matches = xp.squeeze(t_ > 0, -1)
    # The ray hits the 'outside' of the surface instead of the inside.
    # normal_matches = xp.squeeze(dot(n_, a_) < 0, -1)
    # The ray is within the three edges of the triangle, defined using the sign of the dot product.
    within_edges = (dot(n_, cross(e_, p_ - v_)) > 0).prod(axis=-1, dtype=xp.bool)
    # Free the memory used by p
    del p_

    # The matrix of truth values for which all conditions above are satisfied.
    collisions_matrix = within_edges * time_matches # * normal_matches
    # Free the memory used by these boolean matrices
    del within_edges
    del time_matches
    # del normal_matches

    # Determine which rays result in collisions.
    first_collisions_rays_indices  = xp.logical_not(xp.logical_not(collisions_matrix).prod(axis=-1, dtype=xp.bool)).nonzero()[0]

    # Handle the case of no collisions.
    if len(first_collisions_rays_indices) == 0:
        return None, None, None

    # Reduce t_ to just the rays which collide and set any non-collising ray-face pairs to NaN.
    t_ = xp.squeeze(t_, -1)
    t_ = xp.where(collisions_matrix[first_collisions_rays_indices, :], t_[first_collisions_rays_indices, :], xp.nan)

    # Determine which surface the rays which collide do so with first.
    first_collisions_faces_indices = xp.nanargmin(t_, axis=-1)

    # Determine the collision times using the previously determined indices.
    t  = t_[(xp.arange(t_.shape[0]), first_collisions_faces_indices)]

    # Return the time each collision occurs, the index of the ray that caused it and the index of the face it hit.
    return t, first_collisions_rays_indices, first_collisions_faces_indices

