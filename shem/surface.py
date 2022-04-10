import os, sys

import trimesh

import numpy as np
import cupy as cp

from shem.definitions import *

from shem.geometry import vector_dot       as dot
from shem.geometry import vector_cross     as cross
from shem.geometry import vector_normalise as normalise
from shem.geometry import vector_norm      as norm

import shem.ray

# Surface object definition
class Surface:
    def modify_properties(self, properties=None, per_face_props=None):
        '''
        Modify the properties of the surface.
        '''
        if properties is not None:
            self.properties = {k : self.xp.array(v) for k, v in properties.items()}
            if per_face_props is not None:
                # Set properties for each face.
                self.per_face_props  = {k : self.xp.array(v) for k, v in per_face_props.items()}
            else:
                # Define the properties for all faces to be identical.
                self.per_face_props  = None

            if self.per_face_props is not None:
                if self.per_face_props.keys() != self.properties.keys():
                    raise ValueError("per_face_props and properties must be dictionaries with identical keys.")
                if self.xp.array(list(self.per_face_props.values())).shape[1] != self.faces.shape[0]:
                    raise ValueError("The dictionary per_face_group must index lists of indices all of length f (the number of faces).")
        else:
            self.properties = None
            self.per_face_props = None
        return

    def __init__(self, vertices, faces, properties=None, per_face_props=None, xp=np):
        # Sanity checking
        if vertices.shape[1] != 3:
            raise ValueError("vertices must be an n x 3 array; got " + str(vertices.shape))
        if faces.shape[1] != 3:
            raise ValueError("faces must be an n x 3 array; got " + str(faces.shape))

        # Choose whether to run on the CPU or GPU
        self.xp         = xp
        self.vertices   = self.xp.array(vertices)
        self.faces      = self.xp.array(faces, dtype=int)
        self.modify_properties(properties, per_face_props)

        """
        # Define the properties as a dictionary of lists of tensors.
        if properties is not None:
            self.properties = {k : self.xp.array(v) for k, v in properties.items()}
            if per_face_props is not None:
                # Set properties for each face.
                self.per_face_props  = {k : self.xp.array(v) for k, v in per_face_props.items()}

            else:
                # Define the properties for all faces to be identical.
                self.per_face_props  = None

            if self.per_face_props is not None:
                if self.per_face_props.keys() != self.properties.keys():
                    raise ValueError("per_face_props and properties must be dictionaries with identical keys.")
                if xp.array(list(self.per_face_props.values())).shape[1] != self.faces.shape[0]:
                    raise ValueError("The dictionary per_face_group must index lists of indices all of length f (the number of faces).")
        else:
            self.properties = None
            self.per_face_props = None

        """

        # We may as well precalculate these since we will be using them anyway...
        self.triangles  = self._triangles()
        self.edges      = self._edges()
        self.normals    = self._normals()
        self.centroids  = self._centroids()
        return

    def shift(self, v_):
        v = self.xp.array(v_)
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

    def get_property(self, key):
        if self.properties is None:
            return
        if self.per_face_props is not None:
            # Return a list of p x f list of properties with the outer list being a regular Python one (different properties have different dimensions).
            return self.properties[key][self.per_face_props[key]]
        else:
            return self.properties[key]

    def _triangles(self):
        return self.vertices[self.faces]
    
    def _centroids(self):
        return self.triangles.mean(-2)
    
    def face_vector_areas(self):
        return cross(self.edges[..., 0, :], self.edges[..., 1, :]) / 2

    def face_areas(self):
        return norm(self.face_vector_area(), axis=-1)

    def _edges(self):
        return self.xp.array([
            self.triangles[:, 1] - self.triangles[:, 0],
            self.triangles[:, 2] - self.triangles[:, 1],
            self.triangles[:, 0] - self.triangles[:, 2],
        ]).transpose(1,0,2)
    
    def _normals(self):
        return normalise(self.face_vector_areas())


def modify_surface(surface, settings):
    # Get the surface properties.
    properties, per_face_props = settings["sample"]["properties"]["function"](surface.vertices, surface.faces, **settings["sample"]["properties"]["kwargs"])
    # Apply the surface properties
    surface.modify_properties(properties, per_face_props)

    return surface

def load_surface(args, settings):
    '''
    Load the mesh file into a surface object.
    '''
    mesh_file = os.path.join(args.work_dir, args.mesh)
    
    # If necessary, extract the combined meshes from the file. This can be a problem with different file formats.
    try:
        mesh = trimesh.load_mesh(mesh_file).dump(concatenate=True)
    except:
        mesh = trimesh.load_mesh(mesh_file)
    
    # Create the surface object
    surface = shem.surface.Surface(vertices = mesh.vertices,
                                      faces = mesh.faces,
                                         xp = get_xp(args))
     
    # Apply coordinate shifts to the surface. It doesn't make sense to put this in modify_surface since tweaking the shift as a parameter means the surface is shifted repeatedly (not good).
    surface.shift(settings["sample"]["shift"])

    # Apply any necessary modifications
    surface = modify_surface(surface, settings)
    
    return surface
