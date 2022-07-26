# Do this all on the CPU
import numpy as np
import trimesh
import os, sys

from shem.definitions import *

# Vertices of a slab with the top surface in the (x,y,0) plane.
PLATFORM_VERTICES = np.array([
    [-1,-1,-1],
    [ 1,-1,-1],
    [-1, 1,-1],
    [ 1, 1,-1],
    [-1,-1, 0],
    [ 1,-1, 0],
    [-1, 1, 0],
    [ 1, 1, 0],
], dtype=np.float32)

PLATFORM_VERTICES[:, 0:2] /= 2

# Faces of a slab sans the top surface.
PLATFORM_FACES = np.array([
    [0, 2, 1],
    [1, 2, 3],
    [0, 4, 6],
    [0, 6, 2],
    [5, 1, 3],
    [5, 3, 7],
    [3, 2, 7],
    [2, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
], dtype=int)


# TODO: Allow the user to halt the program if the mesh is inadequate.
def inspect_mesh(mesh):
    '''
    Accepts a mesh, recieves user input and returns whether the mesh is acceptable.
    '''
    mesh.show()
    return True

def convert_triangles_to_mesh(mesh_triangles):
    '''
    Accepts an N x 3 x 3 numpy array and returns the corresponding mesh object.
    '''
    mesh = trimesh.Trimesh(use_embree=False, **trimesh.triangles.to_kwargs(mesh_triangles))
    return mesh

def create_flat_triangles(w, h):
    '''
    Accepts the width and height of a slab.
    Returns an N x 3 x 3 numpy array of triangles representing a slab of specified dimensions.
    '''
    surface_faces = np.array([
        [5, 7, 6],
        [5, 6, 4],
    ])

    platform_vertices_ = PLATFORM_VERTICES
    platform_vertices_[:, 0] *= w
    platform_vertices_[:, 1] *= w
    platform_vertices_[:, 2] *= h

    flat_triangles = np.vstack((
        platform_vertices_[PLATFORM_FACES],
        platform_vertices_[surface_faces],
    ))

    return flat_triangles

def create_sphere_triangles(w, h, r, iterations=3):
    '''
    Create a mesh comprised of a sphere radius r atop a flat platform dimensions w x w x h.
    Generates the sphere using a recursive tesselation algorithm, with 3 layers of recursion by default.
    '''

    # We will only be using this function in this limitted scope, so this is fine.
    def tesselate_sphere_triangles(sphere_triangles):
        '''
        Accepts a numpy array of vertices on the surface of a sphere.
        Returns an array four times larger by taking the edge midpoints of each vertex triplet and using them to create four new triangles.
        '''
        # Create the output array
        new_sphere_triangles = np.empty((4*sphere_triangles.shape[0], 3, 3), dtype=float)
        vertex_pairs = np.array([
            [0,1],
            [1,2],
            [2,0]
        ])

        # Calculate the new vertex triplets
        for i in range(sphere_triangles.shape[0]):
            for j in range(3):
                new_sphere_triangles[4*i+j] = np.array([
                    sphere_triangles[i][j],
                    sphere_triangles[i][vertex_pairs[(j+2)%3]].mean(0),
                    sphere_triangles[i][vertex_pairs[(j+3)%3]].mean(0),
                ])
            
            # Central triangle calculated seperately.
            new_sphere_triangles[4*i+3] = np.array([
                sphere_triangles[i][vertex_pairs[2]].mean(0),
                sphere_triangles[i][vertex_pairs[1]].mean(0),
                sphere_triangles[i][vertex_pairs[0]].mean(0),
            ])

        new_sphere_triangles = new_sphere_triangles / np.linalg.norm(new_sphere_triangles, axis=-1).reshape(-1,3,1)
        # new_sphere_triangles = (new_sphere_triangles.T / np.linalg.norm(new_sphere_triangles.T, axis=0)).T
        return new_sphere_triangles

    # Start the tesselation process with an octohedron.
    octahedron_vertices = np.array([
        [ 0, 0,-1],
        [ 1, 0, 0],
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0, 1],
    ], dtype=np.float32)
    
    # I have swapped the order of the face indices here around so the normals point in the correct direction.
    octahedron_faces = np.array([
        [0, 2, 1],
        [0, 3, 2],
        [0, 4, 3],
        [0, 1, 4],
        [5, 1, 2],
        [5, 2, 3],
        [5, 3, 4],
        [5, 4, 1],
    ])[:, ::-1]

    
    # Initialise with the octohedron.
    sphere_triangles_ = octahedron_vertices[octahedron_faces]

    # Repeat the tesselation process.
    for i in range(iterations):
        sphere_triangles_ = tesselate_sphere_triangles(sphere_triangles_)

    # Shift up so that the bottom of the sphere just touches the slab.
    sphere_triangles_[:, :, 2] += 1.0
    # Adjust the sphere to the appropriate radius
    sphere_triangles_[:, :, :] *= r

    # Platform vertices plus the point at base of the sphere.
    platform_vertices_ = np.vstack((
        PLATFORM_VERTICES,
        [0, 0, 0],
    ))
    
    # Adjust to the correct dimensions.
    platform_vertices_[:, 0] *= w
    platform_vertices_[:, 1] *= w
    platform_vertices_[:, 2] *= h

    # Faces connecting the sphere to the slab.
    interface_faces = np.array([
        [4, 8, 6],
        [5, 8, 4],
        [7, 8, 5],
        [6, 8, 7],
    ])

    sphere_triangles = np.vstack((
        platform_vertices_[PLATFORM_FACES],
        platform_vertices_[interface_faces],
        sphere_triangles_,
    ))
 
    return sphere_triangles

def create_trenches_triangles(w, h, t_w, t_h, t_d):
    '''
    Accepts the width and height of a slab, the number of trenches and the relative width and height relative to the maximum possible and the depths of each trench.
    Returns an N x 3 x 3 numpy array of triangles representing a slab of specified dimensions.
    '''
    assert t_w < 1.0
    assert t_h < 1.0
    for d in t_d:
        assert d < 1.0

    # Number of trenches
    n = len(t_d)

    trench_vertices = np.array([
        np.array([2*i+1-n, 0, 0]) + np.array([
            [-t_w, -t_h,    0],
            [+t_w, -t_h,    0],
            [+t_w, +t_h,    0],
            [-t_w, +t_h,    0],
            [-t_w, -t_h, -2*d],
            [+t_w, -t_h, -2*d],
            [+t_w, +t_h, -2*d],
            [-t_w, +t_h, -2*d],
        ]) for i, d in enumerate(t_d)
    ])

    # Divide by n+1 to allow space at the edges
    trench_vertices[..., 0] /= n+1

    # Faces within the trenches
    within_trench_faces = np.array([
        # Sides
        [0,5,4],
        [0,1,5],
        [1,6,5],
        [1,2,6],
        [2,7,6],
        [2,3,7],
        [3,4,7],
        [3,0,4],
        #Bottom
        [4,5,6],
        [6,7,4],
    ])


    # Define the trench width relative to the first and final trenches
    t_w_ = trench_vertices[-1,1,0]

    # Triangles outlining the gap between trenches and flat surface
    flat_surface_triangles = np.array([
        # Triangles to the sides of trenches
        [[-1, -t_h, 0], [-1   , +t_h, 0], [-t_w_, -t_h, 0]][::-1],
        [[-1, +t_h, 0], [-t_w_, -t_h, 0], [-t_w_, +t_h, 0]],
        [[+1, -t_h, 0], [+1   , +t_h, 0], [+t_w_, -t_h, 0]],
        [[+1, +t_h, 0], [+t_w_, -t_h, 0], [+t_w_, +t_h, 0]][::-1],
        # Triangles above and below trenches
        [[-1, -1, 0], [+1,   -1, 0], [+1, -t_h, 0]],
        [[-1, -1, 0], [+1, -t_h, 0], [-1, -t_h, 0]],
        [[-1, +1, 0], [+1,   +1, 0], [+1, +t_h, 0]][::-1],
        [[-1, +1, 0], [+1, +t_h, 0], [-1, +t_h, 0]][::-1],
    ])

    # Surface between trenches
    between_surface_vertices = np.stack((
        trench_vertices[:-1, [1,2]],
        trench_vertices[+1:, [0,3]],
    )).transpose(1,0,2,3).reshape(-1,4,3)

    between_surface_faces = np.array([
        [0,3,1],
        [0,2,3],
    ])

    between_surface_triangles = between_surface_vertices[:, between_surface_faces, :]

    flat_triangles = np.vstack((
        PLATFORM_VERTICES[PLATFORM_FACES],
        np.vstack(trench_vertices[:, within_trench_faces, :]) / 2,
        flat_surface_triangles / 2,
        np.vstack(between_surface_triangles) / 2,
    ))

    # Rescale the mesh
    flat_triangles[..., 0] *= w
    flat_triangles[..., 1] *= w
    flat_triangles[..., 2] *= h

    return flat_triangles
