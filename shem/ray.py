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

def diffuse(a, n):
    xp = cp.get_array_module(a, n)
    # Generate random directions
    directions = xp.random.rand(a.shape[0], 2)
    directions[:, 0] *= 2*xp.pi
    directions[:, 1] *= xp.pi/2
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


def detect_collisions_matrix_method(r, s):
    xp = cp.get_array_module(r)

    a = r[0]
    b = r[1]
    n = s.normals()
    e = s.edges()
    v = s.triangles()
    c = s.centroids()

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
    normal_matches = xp.squeeze(dot(n_, a_) < 0, -1)
    # The ray is within the three edges of the triangle, defined using the sign of the dot product.
    within_edges = (dot(n_, cross(e_, p_ - v_)) > 0).prod(axis=-1, dtype=xp.bool)

    # The matrix of truth values for which all conditions above are satisfied.
    collisions_matrix = within_edges * time_matches * normal_matches

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

# TODO: Try and get this working...
def detect_collisions_index_method(r, s):
    xp = cp.get_array_module(r)
   
    a = r[0]
    b = r[1]
    n = s.normals()
    e = s.edges()
    v = s.triangles()
    c = s.centroids()

    r_dim = r.shape[1]
    f_dim = s.faces.shape[0]

    # Reshape arrays for minimally painful broadcasting.
    n_ = n.reshape(1,     f_dim, 1, 3)
    e_ = e.reshape(1,     f_dim, 3, 3)
    v_ = v.reshape(1,     f_dim, 3, 3)
    c_ = c.reshape(1,     f_dim, 1, 3)
    a_ = a.reshape(r_dim, 1,     1, 3)
    b_ = b.reshape(r_dim, 1,     1, 3)

    # TODO: This uses way too much memory and computational time.
    # Wouldn't it be much better if we only calculated the time and, more imporatantly, intersection points only for the set of rays fulfilling the previously outlined criteria?
    # The number of plane intersection points scales massively...
    # For now we can reduce this by iterating over the block size.
    
    # Indices where ray hits the 'outside' of the surface instead of the inside.
    normal_matches = xp.squeeze(dot(n_, a_) < 0, -1)

    # Take only rays where the normal matches.
    normal_matches_indices = normal_matches.nonzero()
    rays_slice_n  = (normal_matches_indices[0], xp.zeros_like(normal_matches_indices[0]))
    faces_slice_n = (xp.zeros_like(normal_matches_indices[1]), normal_matches_indices[1])
    n_ = n_[faces_slice_n]
    e_ = e_[faces_slice_n]
    v_ = v_[faces_slice_n]
    c_ = c_[faces_slice_n]
    a_ = a_[rays_slice_n]
    b_ = b_[rays_slice_n]

    # Calculate the time each ray intersects with each plane and expand for broadcasting.
    t_ = intersects_plane_time(a_, b_, n_, c_)

    # Indices in normal_matches_indices for which t_ > 0.
    time_matches_indices = (t_ > 0).nonzero()[0]
    n_ = n_[time_matches_indices]
    e_ = e_[time_matches_indices]
    v_ = v_[time_matches_indices]
    c_ = c_[time_matches_indices]
    t_ = t_[time_matches_indices]
    a_ = a_[time_matches_indices]
    b_ = b_[time_matches_indices]

    # Point each ray intersects with each plane.
    p_ = parameterise(a_, b_, t_)
    
    # The ray is within the three edges of the triangle, defined using the sign of the dot product.
    within_edges = (dot(n_, cross(e_, p_ - v_)) > 0).prod(axis=-1, dtype=xp.bool)

    # Indices in time_matches_indices for which the point lies on the triangle.
    within_edges_indices = within_edges.nonzero()[0]

    t_ = t_[within_edges_indices]

    # The ray and face indices which lead to collisions.
    collisions_rays_indices  = normal_matches_indices[0][time_matches_indices][within_edges_indices]
    collisions_faces_indices = normal_matches_indices[1][time_matches_indices][within_edges_indices]

    # Determine which surface the rays which collide do so with first.
    first_collisions_rays_indices  = xp.unique(collisions_rays_indices)
    first_collisions_faces_indices = xp.empty_like(collisions_rays_indices)
    # Set all times to be infinity initially
    t = xp.full_like(collisions_rays_indices, xp.inf, dtype=xp.float32)

    # Iterate over the intersection times.
    # For surfaces in SHeM we expect a ray to collide with perhaps 2 faces - front and back - so the CPU processing time should be neglible.
    # A surface with an interior full of cavities might be more problematic.
    for i in range(t_.shape[0]):
        ray  = collisions_rays_indices[i]
        face = collisions_faces_indices[i]
        # If this intersection time is less than what we have already, set the output to it.
        if t_[i] < t[ray]:
            t[ray] = t[i]
            first_collisions_faces_indices[ray] = collisions_faces_indices[ray]

    
    # Handle the case of no collisions.
    if len(first_collisions_rays_indices) == 0:
        return None, None, None

    # Return the time each collision occurs, the index of the ray that caused it and the index of the face it hit.
    return t, first_collisions_rays_indices, first_collisions_faces_indices


def detect_collisions(r, s, method='matrix'):
    if method != 'matrix' and method != 'index':
        return None, None, None

    if method == 'matrix':
        return detect_collisions_matrix_method(r,s)

    if method == 'index':
        return detect_collisions_index_method(r,s)
