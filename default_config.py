import shem.definitions
import shem.geometry
import shem.mesh
import shem.ray
import shem.scattering
import shem.source

import numpy as np
try:
    import cupy  as cp
except:
    pass

# Override the mesh and config hashes stored in the database.
override_hashes = True

# A starting point so that the algorithm needs less time to calibrate.
# If this is too large the simulation will crash
max_rays_scan_init = 16500

# The maximum number of times to scan over the coordinates.
# Set to -1 run run indefinitely.
max_scans = 2

# Sample Parameters
## Coordinate Shifts
x_shift = 0.0
y_shift = 0.0
z_shift = 0.0

# Scan Parameters
cart_steps = 2**1
width = 0.5
x     = np.linspace(-width/2, +width/2, cart_steps)
y     = np.linspace(-width/2, +width/2, cart_steps)

polar_steps = 2**8
cart_polar_steps = 2**1
angle = np.radians(1)
x_polar = np.linspace(-width/2, +width/2, cart_polar_steps)
y_polar = np.linspace(-width/2, +width/2, cart_polar_steps)
theta   = np.linspace(-angle/2, +angle/2,      polar_steps)
phi     = np.linspace(-angle/2, +angle/2,      polar_steps)

# Scan over the x and y indices at (theta, phi) = (0, 0)
cart_scan_single = np.stack(tuple(map(np.ravel, np.meshgrid(x,       y,           0,   0))))
# Scan over the theta and phi indices for multiple values of (x,y)
polar_scan_multi = np.stack(tuple(map(np.ravel, np.meshgrid(x_polar, y_polar, theta, phi))))

coordinates = np.hstack((
    cart_scan_single,
    # polar_scan_multi,
))

# Surface properties.
# For now, we define this here as a function of the vertices and faces of the mesh.
# There is probably a nicer way to do it by embedding the information in the glb file format but I'm not sure that would even work for what I want, hence, this approach.
def surface_properties_uniform(v, f):
    # property keys. In this case we define three separate reciprocal lattice vectors and a scalar.
    # We could just as easily put all the reciprocal lattice vectors into one array but we don't here for clarity.
    keys = ['x', 'b1', 'b2', 'b3']
    properties_arr = [
            0.5,
            np.array([1,0,0]),
            np.array([0,1,0]),
            np.array([0,0,1]),
    ]
    
    properties = dict(zip(keys, properties_arr))
    per_face_props = None

    # How to assign different properties to each face.
    
    """
    properties_arr = [
            np.arange(2,11) / 10,
            np.array([[1,0,0],[2,0,0]]),
            np.array([[0,1,0],[0,2,0],[0,3,0]]),
            np.array([[0,1,0]]),
    ]
    properties = dict(zip(keys, properties_arr))
    per_face_props_arr = [
        [0,1,4,7],
        [0,1,1,0],
        [1,1,2,0],
        [0,0,0,0],
    ]
    """
    return properties, per_face_props

# How to assign different properties to each face.
def surface_properties_random(v, f):
    # We will asign each face a random set of properties.
    # Another possible example is to asign every face below a point a set of 'stage' properties, above that as 'deposited material' and far above that as 'pinhole plate'.
    # This example is useful for if e.g. there is some discontinuity in the crystal structure.
    # Mostly, it's just illustrative.

    keys = ['x', 'b1', 'b2', 'b3']

    # The different properties each face may have
    properties_arr = [
            np.arange(2,11) / 10,
            np.array([[1,0,0],[2,0,0]]),
            np.array([[0,1,0],[0,2,0],[0,3,0]]),
            np.array([[0,1,0]]),
    ]
    properties = dict(zip(keys, properties_arr))

    # Asign each face random properties from those provided
    per_face_props_arr = [np.random.randint(0, p.shape[0], f.shape[0]) for k,p in properties.items()]
    per_face_props = dict(zip(keys, per_face_props_arr))
    
    return properties, per_face_props

surface_properties = surface_properties_uniform

# Scattering
max_scatters = 255 # Maximum number of times rays can scatter. Must be between 0 and 255.

# Custom scattering function.
def my_specular(a, f, s):
    '''
    All user defined scattering functions should accept at least the following arguments:
    a is the incoming direction vector,
    s is the surface object,
    f is the list of face indices corresponding to each ray (can be used to extract the properties of the surface).

    Additional arguments may be provided in the scattering function specified by the simulation.
    It is good practice to provide default values for these scattering functions.
    '''

    # We don't need to have this function work with NumPy or CuPy, but this is good practice.
    xp = cp.get_array_module(a, f, s.vertices)
    # Get the normals from the surface object and the face indices
    n = s.normals[f]
    return shem.ray.reflect(a, n)


"""
The scattering function is described by a dictionary indexed by the name of the function.
Its values are three element tuples consisting of the function itself, the relative strength of the function and a dictionary of kwargs supplied to the function.
"""
scattering_function = {
    "Perfect Diffuse Scattering"  : ( shem.ray.perfect_diffuse,  10, {
    }),
    "Perfect Specular Scattering" : ( shem.ray.perfect_specular, 90, {
    }),
}

# As we might expect, you can provide multiple scattering functions in an array.
scattering_functions = np.array([
    scattering_function
])

# Source
source_angle    = np.degrees(1) # Trace rays in a cone of diameter 1 degree from the point source.
#source_angle    = 0
source_location = shem.geometry.polar2cart([1.0, 180.0, 45.0], radians=False)

############
# Analysis #
############

detector_location = shem.geometry.polar2cart(np.array([
    [1.0, 0.0, 45.0],
    [1.0, 0.0, 45.0],
    [1.0, 0.0, 45.0],
    [1.0, 0.0, 45.0],
    [1.0, 0.0, 45.0],
    [1.0, 0.0, 45.0],
    [1.0, 0.0, 45.0],
    [1.0, 0.0, 45.0],
]))

detector_radius = np.arange(1, detector_location.shape[0]+1) / 100

plots = {
    "Standard X-Y" : {
        'axes' : 'xy',
    },
    "Standard Theta-Phi" : {
        'axes' : 'tp',
    },
}




