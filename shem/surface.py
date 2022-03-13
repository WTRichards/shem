import os, sys

import numpy as np
import cupy as cp

from shem.definitions import *

from shem.geometry import vector_dot       as dot
from shem.geometry import vector_cross     as cross
from shem.geometry import vector_normalise as normalise
from shem.geometry import vector_norm      as norm

import shem.ray

# Surface object definition
class SurfaceGPU:
    def __init__(self, vertices, faces):
        if vertices.shape[1] != 3:
            raise ValueError("vertices must be an n x 3 array; got " + str(vertices.shape))
        if faces.shape[1] != 3:
            raise ValueError("faces must be an n x 3 array; got " + str(faces.shape))
        self.vertices = cp.array(vertices)
        self.faces    = cp.array(faces, dtype=int)
        # Raise the surface by a small amount to avoid problems with points not being deteced on geometrically peerfect edges.
        self.shift_z(DELTA)
        return

    def shift(self, v_):
        v = cp.array(v_)
        self.vertices += v
        return
    
    def shift_x(self, x):
        self.vertices[:, X] += x
        return
    
    def shift_y(self, y):
        self.vertices[:, Y] += y
        return
    
    def shift_z(self, z):
        self.vertices[:, Z] += z
        return

    def triangles(self):
        return self.vertices[self.faces]
    
    def centroids(self):
        return self.triangles().mean(-2)

    def edges(self):
        faces    = self.faces
        vertices = self.vertices
        
        triangles = vertices[faces]

        return cp.array([
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ]).transpose(1,0,2)
    
    def face_vector_areas(self):
        edges = self.edges()
        return cross(edges[..., 0, :], edges[..., 1, :]) / 2

    def face_areas(self):
        return norm(self.face_vector_area(), axis=-1)

    def normals(self):
        return normalise(self.face_vector_areas())


class SurfaceCPU:
    def __init__(self, vertices, faces):
        if vertices.shape[1] != 3:
            raise ValueError("vertices must be an n x 3 array; got " + str(vertices.shape))
        if faces.shape[1] != 3:
            raise ValueError("faces must be an n x 3 array; got " + str(faces.shape))
        self.vertices = np.array(vertices)
        self.faces    = np.array(faces, dtype=int)
        # Raise the surface by a small amount to avoid problems with points not being deteced on geometrically peerfect edges.
        self.shift_z(DELTA)
        return

    def shift(self, v_):
        v = np.array(v_)
        self.vertices += v
        return
    
    def shift_x(self, x):
        self.vertices[:, X] += x
        return
    
    def shift_y(self, y):
        self.vertices[:, Y] += y
        return
    
    def shift_z(self, z):
        self.vertices[:, Z] += z
        return

    def triangles(self):
        return self.vertices[self.faces]
    
    def centroids(self):
        return self.triangles().mean(-2)

    def edges(self):
        faces    = self.faces
        vertices = self.vertices
        
        triangles = vertices[faces]

        return np.array([
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ]).transpose(1,0,2)
    
    def face_vector_areas(self):
        edges = self.edges()
        return cross(edges[..., 0, :], edges[..., 1, :]) / 2

    def face_areas(self):
        return norm(self.face_vector_area(), axis=-1)

    def normals(self):
        return normalise(self.face_vector_areas())
