import shem.definitions
import shem.geometry
import shem.mesh
import shem.ray
import shem.scattering
import shem.source

# You shouldn't need GPU acceleration for setting up the config file.
import numpy as np


# TODO: Document the variables which actually do things here. It would be nice if we had a stable API so I could tell you which variables actually mattered but everything is still in flux...
# For now, there will just be a # USED above each variable the program uses, which hopefully should be clear enough...
"""
This is the default configuration file for the shem simulator.
It is loaded as a Python module so you can put any code here and it will be run.
Always be mindful of the code you choose to run.
"""


########################
# Operating Parameters #
########################
"""
The values defined here should not affect the result of the simulation, only how it is run.
Adjusting these may cause crashes by e.g. causing the system to run out of memory, using a config file which does not match a database.
You shouldn't need to change these by and large.
If you are running on the GPU the program will figure out how much memory it can use automatically (it may take a few minutes in the worst case).
You should really only be running on the CPU for debugging.
"""

# Override the mesh and config hashes stored in the database.
# USED
override_hashes = True

# A starting point so that the algorithm needs less time to calibrate how much memory it can use.
# If this is too large the program may just crash.
# USED
max_rays_scan_init = 10000

# The maximum number of times to scan over the coordinates.
# Set to -1 run indefinitely.
# USED
max_scans = 1

# Maximum number of times rays can scatter off the surface.
# Since this is stored as an 8 bit integer it must be between 0 and 255.
# For most purposes, 255 is effectively unlimited.
# USED
max_scatters = 255


##########
# Sample #
##########
"""
Here, we define the various modifications we wish to make to the sample.
One obvious option is that we wish to perform some kind of coordinate transformation.
We may also wish to ascribe properties to the sample, either per face or as a whole.
Ideally, we would be able to create a file containing information on the sample topology (triangle mesh) and its general properties.
Unfortunately, existing formats for computer graphics are not suited for the kind of physics we are interested in.
This is a workaround which ascribes properties to the sample based on its faces and vertices.
These properties may be arbitrary tensors and can be used in scattering functions.
A trivial use of this would be to consider 2 z values z0, z1.
If the face is below z=z1 then it has the properties of graphite, if z0 < z < z1 then it has the properties of some silicate bead, if z > z1 then it has the properties of the pinhole plate.
Again, these may be arbitrary tensors.
"""

# Shift the sample
# USED
x_shift = 0.0
# USED
y_shift = 0.0
# USED
z_shift = 0.0

# TODO: Add the ability to rotate the sample. Rotations don't generally commute and the sample may not be centered at the origin so this could be tricky to do intuitively...

# Define the properties of the surface as a single dictionary of tensors
def surface_properties_uniform(v, f):
    # The dictionary is indexed by the name of the tensor.
    # In this case we define three separate reciprocal lattice vectors b1, b2, b3 and a scalar x.
    # We could just as easily put all the reciprocal lattice vectors into a single 3 x 3 tensor.
    keys = ['x', 'b1', 'b2', 'b3']
    properties_arr = [
            0.5,
            np.array([1,0,0]),
            np.array([0,1,0]),
            np.array([0,0,1]),
    ]
    
    properties = dict(zip(keys, properties_arr))
    per_face_props = None

    return properties, per_face_props

# Assign each face a different (random) set of properties.
# This isn't particularly useful as written but it is illustrative.
def surface_properties_random(v, f):

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

# Set the surface properties function to the uniform one defined above.
# USED
surface_properties = surface_properties_uniform


##########
# Source #
##########
"""
Here, we define the properties of the helium source.
We take the reasonable assumption of a point source.
The simulation will trace rays at random with angle equal to or less than source_angle.
Because we record the theta and phi components of these rays relative to a 'perfect' source beam we can convolve our result with the source distribution after we have traced and detected the rays.
This is computationally more efficient and so allows us to quite easily tweak the source distribution.
The downside is that we will always be applying a uniform cone window function.
"""
# Trace rays in a 1 degree cone by default
# USED
source_angle    = np.radians(1)
# Place the source at a 45 degree angle
# USED
source_location = shem.geometry.polar2cart([1.0, 180.0, 45.0], radians=False)

# Custom source function
def my_uniform(theta, phi, **kwargs):
    '''
    All user defined scattering functions should accept the following arguments:
    theta is the azimuthal angle made with respect to a perfect ray,
    phi   is the     polar angle made with respect to a perfect ray,
    **kwargs defines a number of keyword arguments. This allows us to tune a function's behaviour without creating an entirely new function. It also lets us (hopefully) attempt to fit for various parameters.
    '''
    # We don't need to have this function work with both CPU and GPU explicitly (you can usually just use standard Python operators and it will understand) but it is good practice to define xp if using functions from NumPy.
    # CuPy is largely identical to NumPy so in most cases you can just replace np with xp.
    xp = cp.get_array_module(theta, phi)
    return xp.ones_like(theta)


"""
The source function is defined by a dictionary indexed by the name of the function.
Its values are three element tuples consisting of the function itself, the relative strength of the function and a dictionary of kwargs supplied to the function.
{"function name": (function, relative strength, kwargs)}
"""
sigma1 = 1.0
sigma2 = 0.5
# We will create a double Gaussian by using the Gaussian function twice with different relative strengths and standard deviations.
# USED
source_function = {
    "Uniform Source"  : (           my_uniform, 20, {
    }),
    "First Gaussian"  : ( shem.source.gaussian, 40, {
        "sigma" : sigma1,
    }),
    "Second Gaussian" : ( shem.source.gaussian, 40, {
        "sigma" : sigma2,
    }),
}


############
# Scanning #
############
"""
Define how the surface scan will be conducted.
These defaults are small but sensible and prove the simulation works.
We will we define the coordinates we scan over as a 4 x n NumPy array.
We can specify x, y, theta and phi coordinates.
Here, we define theta as the azimuthal angular displacement and phi as the polar angular displacement relative to the location of the source.
While in reality the sample is moved about on a stage it is computationally more efficient to move the source and keep the sample static.
"""

# A scan over the x and y indices at (theta, phi) = (0, 0)
cart_steps = 2
width = 0.5
x = np.linspace(-width/2, +width/2, cart_steps)
y = np.linspace(-width/2, +width/2, cart_steps)
cart_scan_single = np.stack(tuple(map(np.ravel, np.meshgrid(x,       y,          0,  0))))

# Scan over the theta and phi indices for multiple values of (x,y)
polar_steps = 2
cart_polar_steps = 2
angle = np.radians(30)
x_polar = np.linspace(-width/2, +width/2, cart_polar_steps)
y_polar = np.linspace(-width/2, +width/2, cart_polar_steps)
theta   = np.linspace(-angle/2, +angle/2, polar_steps)
phi     = np.linspace(-angle/2, +angle/2, polar_steps)
polar_scan_multi = np.stack(tuple(map(np.ravel, np.meshgrid(x_polar, y_polar, theta, phi))))

# Combine multiple scans into a single 4 x n array.
# USED
coordinates = np.hstack((
    cart_scan_single,
    polar_scan_multi,
))


############
# Plotting #
############
"""
Once we have traced the rays, detected them and applied the source function, we need to create an intensity plot we can view and analyse.
To do this, we define the plots we wish to make as a dictionary.
We will have to specify which coordinate indices we wish to use to make the plots.
"""

# The indices of the different scans.
# NOTE: You will need to manage these yourself if you want to create any figures from the data you will have generated.
current_index = 0
cart_scan_single_indices = np.arange(current_index, current_index + cart_scan_single.shape[1])
current_index += cart_scan_single.shape[1]
polar_scan_multi_indices = np.arange(current_index, current_index + polar_scan_multi.shape[1])
current_index += polar_scan_multi.shape[1]

x, y, theta, phi = 0, 1, 2, 3
# USED
plots = [
    {
        "image"   : False,                         # Output an image rather than a full figure. Useful if the images will be used for more plots. Defaults to False.
        "flip"    : True,                          # Flip the y axis. This is necessary if we want to see the image we expect. See the camera obscura. Defaults to True.
        "title"   : "Standard X-Y Intensity Plot", # Title of the plot
        "type"    : "xy intensity",                # Type of the plot. Determines how the rest of the dictionary is parsed.
        "indices" : cart_scan_single_indices,      # For an "xy intensity" plot we need to know which indices we will be plotting.
        "x"       :  x,                            # The x coordinate. Despite the name, an "xy intensity" plot can track any coordinate index
        "x_name"  : "x",                           # The x axis title. We can use this to e.g. include units
        "y"       :  y,                            # Identical to the x axis
        "y_name"  : "y",                           # Identical to the x axis
        "z"       : "signal",                      # We specify which field from the many tables we wish to plot. Here, we will just look at the signal but we could look at other properties like n_scat.
    },
]






############
# Detector #
############
"""
Here, we define the properties of the detector.
This is somewhat barebones and we might like to, say, define a detector slit instead of a pinhole but it works well enough.
"""

# Place the detector at a 45 degree angle opposite the source.
# USED
detector_location = shem.geometry.polar2cart([1.0, 0.0, 45.0], radians=False)
# Define the radius of the detector
# USED
detector_radius = np.arange(1, detector_location.shape[0]+1) / 100
# Define the normal of the surface the detector lies in.
# USED
detector_normal = - shem.geometry.vector_normalise(detector_location)


##############
# Scattering #
##############
"""
Here, we define the scattering properties of the sample.
We define the scattering function which determines how helium atoms scatter off the surface as a superposition of multiple scattering functions.
We can (hopefully) use this form to iteratively solve for the scattering distribution used in a particular image.
"""

# Custom scattering function.
def my_specular(a, f, s, **kwargs):
    '''
    All user defined scattering functions should accept the following arguments:
    a is the incoming direction vector,
    s is the surface object,
    f is the list of face indices corresponding to each ray (can be used to extract the properties of the surface).
    **kwargs defines a number of keyword arguments. This allows us to tune a function's behaviour without creating an entirely new function. It also lets us (hopefully) attempt to fit for various parameters.
    '''
    xp = cp.get_array_module(a, f, s.vertices)
    # Get the normals from the surface object and the face indices
    n = s.normals[f]
    return shem.ray.reflect(a, n)


"""
The scattering function is defined by a dictionary indexed by the name of the function.
Its values are three element tuples consisting of the function itself, the relative strength of the function and a dictionary of kwargs supplied to the function.
{"function name": (function, relative strength, kwargs)}
"""
# USED
scattering_function = {
    "Perfect Diffuse Scattering"  : ( shem.scattering.perfect_diffuse,  10, {
    }),
    "Perfect Specular Scattering" : ( shem.scattering.perfect_specular, 90, {
    }),
}




