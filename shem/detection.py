import numpy as np
import cupy  as cp

import shem.geometry
import shem.ray

from shem.definitions import *

###############################
# Surface Collision Detection #
###############################
def detect_surface_collisions(r, s, match_normal=False):
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
    t_ = shem.ray.intersects_plane_time(a_, b_, n_, c_)

    # Point each ray intersects with each plane.
    p_ = shem.ray.parameterise(a_, b_, t_)

    # The rays intersect at time greater than zero.
    time_matches = xp.squeeze(t_ > 0, -1)
    
    # The ray is within the three edges of the triangle, defined using the sign of the dot product.
    within_edges = (shem.geometry.vector_dot(n_, shem.geometry.vector_cross(e_, p_ - v_)) > 0).prod(axis=-1, dtype=xp.bool)
    
    # Free the memory used by p
    del p_

    # The matrix of truth values for which all conditions above are satisfied.
    collisions_matrix = within_edges * time_matches
    # Free the memory used by these boolean matrices
    del within_edges
    del time_matches

    if match_normal:
        # The ray hits the 'outside' of the surface instead of the inside.
        normal_matches = xp.squeeze(shem.geometry.vector_dot(n_, a_) < 0, -1)
        collisions_matrix *= normal_matches
        del normal_matches


    # TODO: See if we can use np.ufunc.reduce with the where argument to make this less circumspect. It works but is confusing...

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

##################
# Detector Types #
##################

def detect_circle(rays, settings, coordinate_indices):
    '''
    Determine the rays detected by a circular aperture radius r, normal n at d.
    '''
    xp = cp.get_array_module(rays, settings, coordinate_indices)
    
    detector_location = xp.array(settings["detector"]["location"])
    n = xp.array(settings["detector"]["normal"])
    r = settings["detector"]["radius"]
    coordinates = xp.array(settings["meta"]["coordinates"])
    
    a = rays[0]
    b = rays[1]


    # Transform polar coordinates TODO: Check that this is the correct transformation. We need to swap the sign on the polar angle to track the detector properly
    angled_detector_locations = shem.geometry.polar2cart( shem.geometry.cart2polar(detector_location) + xp.array([xp.zeros(rays.shape[1]), -coordinates[2][coordinate_indices], coordinates[3][coordinate_indices]]).T )

    # Transform Cartesian coordinates
    d = angled_detector_locations + xp.array([coordinates[0][coordinate_indices], coordinates[1][coordinate_indices], xp.zeros(rays.shape[1])]).T

    # Find the time the ray intersects the plane of the circle.
    t = shem.ray.intersects_plane_time(a, b, n, d)
    
    # Find the point at which the ray lies in that plane
    p = shem.ray.parameterise(a, b, t)

    # The ray is detected at time greater than zero.
    time_matches = t > 0
    # We no longer need t, so free the memory
    del t

    # The ray hits the front face of the detector instead of the rear - not usually a problem.
    #normal_matches = shem.geometry.vector_dot(n, a) < 0
    
    # The ray is within a distance r of the center of the detector.
    within_edges = shem.geometry.vector_norm(p - d) < r
    
    # We no longer need p, so free the memory
    del p

    # The truth vector for when all above conditions are satisfied
    detected = within_edges * time_matches # * normal_matches

    return detected

def detect(rays, settings, coordinate_indices):
    xp = cp.get_array_module(rays, coordinate_indices)
    detector_type = settings["detector"]["type"]

    if detector_type == "circle":
        return detect_circle(rays, settings, coordinate_indices)

    return
