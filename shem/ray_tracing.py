import numpy as np
import trimesh

def intersects_location(rays, mesh, use_gpu):
    '''
    Accepts:
        rays - A 2 x ? x 3 array. It contains the origin and direction vectors of a set of rays.
        mesh - A trimesh object. It contains the mesh of interest.
    Returns:
        A 2 x ? NumPy array containing the indices [ray, face] of successful collisions.
    '''

    # Use trimesh's support for embree - a highly optimised ray tracing library for the CPU.
    # Also, an excellent comparison and much easier to use.
   
    if rays.ndim != 3:
        raise ValueError("The rays array must be 2 x ? x 3.")

    if use_gpu:
        return -1
    else:
        return mesh.ray.intersects_location(ray_origins=rays[1], ray_directions=rays[0], multiple_hits=False)


def intersects_any(rays, mesh, use_gpu):
    '''
    Accepts:
        rays - A 2 x ? x 3 array. It contains the origin and direction vectors of a set of rays.
        mesh - A trimesh object. It contains the mesh of interest.
    Returns:
        A 2 x ? NumPy array containing the indices [ray, face] of successful collisions.
    '''

    # Use trimesh's support for embree - a highly optimised ray tracing library for the CPU.
    # Also, an excellent comparison and much easier to use.
   
    if rays.ndim != 3:
        raise ValueError("The rays array must be 2 x ? x 3.")

    if use_gpu:
        return -1
    else:
        return mesh.ray.intersects_any(ray_origins=rays[1], ray_directions=rays[0])

