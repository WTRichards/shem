import shem.definitions
import shem.geometry
import shem.mesh
import shem.ray
import shem.scattering
import shem.source

# You shouldn't need GPU acceleration for setting up the config file.
import numpy as np

"""
ATTENTION:
This is the default configuration file for the shem simulator.
It has been copied from shem/default_configuration.py.
There are variables in here which are used for different purposes or simply as part of the documentation.
The only variable which will be used in the simulation is the settings dictionary (skip to the end).
You can also see the default settings just below it.
If you modify those when using this as a config file they do nothing - they are illustrative.
"""

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

def surface_properties_uniform(v, f, properties=None):
    '''
    This function defines a surface which is completely uniform with the properties supplied.
    '''
    per_face_props = None

    return properties, per_face_props

# Assign each face a different (random) set of properties.
# This isn't particularly useful as written but it is illustrative.
def surface_properties_random(v, f):
    
    # The dictionary is indexed by the name of the tensor.
    # In this case we define three separate reciprocal lattice vectors b1, b2, b3 and a scalar x.
    # We could just as easily put all the reciprocal lattice vectors into a single 3 x 3 tensor.

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
{"function name": { "function" : function, "strength" : absolute_strength, "kwargs" : {kwargs})}
"""
sigma1 = 1.0
sigma2 = 0.5

uniform = {
    "function" : shem.source.uniform,
    "strength" : 1, 
    "kwargs": {
    },
}

gaussian1 = {
    "function" : shem.source.gaussian,
    "strength" : 0.1, 
    "kwargs": {
        "sigma" : np.radians(1.0)
    },
}

gaussian2 = {
    "function" : shem.source.gaussian,
    "strength" : 0.1, 
    "kwargs": {
        "sigma" : np.radians(0.5)
    },
}

# We will create a double Gaussian by using the Gaussian function twice with different relative strengths and standard deviations.
source_function = {
    "Uniform"         : uniform,
    "First Gaussian"  : gaussian1,
    "Second Gaussian" : gaussian2,
}


###############
# Coordinates #
###############
"""
Define how the surface scan will be conducted.
These defaults are small but sensible and prove the simulation works.
We will we define the coordinates we scan over as a 4 x n NumPy array.
We can specify x, y, theta and phi coordinates.
Here, we define theta as the azimuthal angular displacement and phi as the polar angular displacement relative to the location of the source.
While in reality the sample is moved about on a stage it is computationally more efficient to move the source and keep the sample static.
"""

# A scan over the x and y indices at (theta, phi) = (0, 0)
cart_steps = 32
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


##############
# Scattering #
##############
"""
Here, we define the scattering properties of the sample.
We define the scattering function which determines how helium atoms scatter off the surface as a superposition of multiple scattering functions.
We can (hopefully) use this form to iteratively solve for the scattering distribution used in a particular image.
The scattering function is defined by a dictionary indexed by the name of the function.
{"function name": { "function" : function, "strength" : relative_strength, "kwargs" : {kwargs})}
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


diffuse = {
    "function" : shem.scattering.perfect_diffuse,
    "strength" : 50,
    "kwargs"   : {
    },
}

specular = {
    "function" : shem.scattering.perfect_specular,
    "strength" : 50,
    "kwargs"   : {
    },
}

scattering_function = {
    "Perfect Diffuse Scattering"  : diffuse,
    "Perfect Specular Scattering" : specular,
}

################
# Data Display #
################
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
display = {
    # The signal you would expect from an xy scan.
    "signal_xy_intensity_image" : {
    "type"    : "xy intensity",           # Type of the plot. Determines how the rest of the dictionary is parsed.
    "flip"    : True,                     # Flip the image if plotting in x and y. This puts the image the right way up - see the camera obscura.
    "figure"  : False,                    # Output a plot rather than an image. Useful if the images will be used for more plots. Defaults to False.
    "indices" : cart_scan_single_indices, # Which subset of the indices we have simulated we will be plotting.
    "x"       :  x,                       # The x coordinate. Despite the name, an "xy intensity" plot can track any coordinate index
    "x_name"  : "x",                      # The x axis title. We can use this to e.g. include units
    "y"       :  y,                       # Identical to the x axis
    "y_name"  : "y",                      # Identical to the x axis
    "z"       : "signal",                 # We specify which field from the many tables we wish to plot. Here, we will just look at the signal but we could look at other properties like n_scat.
    },
    # The average number of times rays are scattered at each coordinate index.
    "n_s_xy_intensity_image" : {
    "type"    : "xy intensity",           # Type of the plot. Determines how the rest of the dictionary is parsed.
    "flip"    : True,                     # Flip the image if plotting in x and y. This puts the image the right way up - see the camera obscura.
    "figure"  : False,                    # Output a plot rather than an image. Useful if the images will be used for more plots. Defaults to False.
    "indices" : cart_scan_single_indices, # Which subset of the indices we have simulated we will be plotting.
    "x"       :  x,                       # The x coordinate. Despite the name, an "xy intensity" plot can track any coordinate index
    "x_name"  : "x",                      # The x axis title. We can use this to e.g. include units
    "y"       :  y,                       # Identical to the x axis
    "y_name"  : "y",                      # Identical to the x axis
    "z"       : "average number of times scattered",
    },
    # The average number of times rays are scattered at each coordinate index.
    "n_s_detected_xy_intensity_image" : {
    "type"    : "xy intensity",           # Type of the plot. Determines how the rest of the dictionary is parsed.
    "flip"    : True,                     # Flip the image if plotting in x and y. This puts the image the right way up - see the camera obscura.
    "figure"  : False,                    # Output a plot rather than an image. Useful if the images will be used for more plots. Defaults to False.
    "indices" : cart_scan_single_indices, # Which subset of the indices we have simulated we will be plotting.
    "x"       :  x,                       # The x coordinate. Despite the name, an "xy intensity" plot can track any coordinate index
    "x_name"  : "x",                      # The x axis title. We can use this to e.g. include units
    "y"       :  y,                       # Identical to the x axis
    "y_name"  : "y",                      # Identical to the x axis
    "z"       : "average number of times scattered (detected)",
    },
}


###########################
# Optimisation Parameters #
###########################
"""
The template for the parameters the program uses to solve for the settings necessary to get a particular result.
A default value is given in settings and used to test the program.
The solver looks up that particular dictionary element in settings and is able to adjust it and run simulations based on the constraints given in template.
At the moment, only two kinds of parameter are supported:
    1. Scalar parameters in a range give by a tuple (x_min, x_max)
    2. Arbitrary values defined in a list [function_1, function_2, function_3] or ["square", "circle"]
This is sufficient for most optimisation problems.
Ideally we would like to be able to use arbitrary tensors as parameters, compare the "distance" between two tensors, generate a random tensor a specific distance away from a tensor, etc. but that gets complicated very quickly.
We would also like to be able to specify constraints like "this must be a unit vector".
Again, the current specification should be good for most problems.
In fact, you can map a random scalar in some range to a tensor, although you will run into precision issues very quickly...
You could also do some weird stuff using recursion since the optimiser can optimise the template it uses (I don't recommend this but it might be fun).
There is a lot of machine learning research into tuning hyperparameters.
"""

# In this example, we attempt to determine the ratio of specular to diffuse scattering from an arbitrary image, given we know all the other settings.
# This is technically only a single parameter since the overall scattering distribution only cares about the *ratio*.
# With this particular implementation, we need to optimise for the relative strengths of each seperately.

template_simple_scattering = {
    "scattering" : {
        "function" : {
            "Perfect Diffuse Scattering"  : {
                "strength" : (0, 100),
            },
            "Perfect Specular Scattering" : {
                "strength" : (0, 100),
            },
        },
    },
}

# In this example, we solve for the (2D) reciprocal lattice vectors of a sample.
# Note that we still need to define the scattering function such that it can make use of them.
# At the same time, we need to figure out the relative strengths of the different scattering distributions.
# If we already know that most of the scattering will be specular we can bias the RNG by increasing the upper bound since only the relative strengths of the scattering distributions matter.
# The dimensionality of the problem scales incredibly quickly - we have 12 dimensions we can tweak here and that we need to optimise over.
# TODO: Implement simple 2D diffraction so I have some interesting physics stuff to talk about.
            
            # TODO: Implement Simple 2D Diffraction so that this makes sense
            #
            #"Simple 2D Diffraction"       : {
            #    "strength" : (0, 1),
            #    "kwargs"   : {
            #        "sigma_envelope" : (0, 1),
            #        "sigma_peaks"    : (0, 1),
            #    },
            #},

template_reciprocal_lattice = {
    "scattering" : {
        "function" : {
            "Perfect Diffuse Scattering"  : {
                "strength" : [0, 20, 40, 60, 80, 100],
            },
            "Perfect Specular Scattering" : {
                "strength" : (0, 100),
            },
        },
    },
    "sample" : {
        "properties" : {
            "kwargs" : {
                "properties" : {
                    "b1_x" : (0, 1),
                    "b1_y" : (0, 1),
                    "b1_z" : (0, 1),
                    "b2_x" : (0, 1),
                    "b2_y" : (0, 1),
                    "b2_z" : (0, 1),
                },
            },
        },
    },
}


####################
# Default Settings #
####################
"""
ATTENTION:
These are the default settings.
If you do not supply a setting in settings, these are the values that will be used.
You can adjust the settings but you cannot adjust the default settings.
This is just a copy of them which has no effect on the program.
THESE __DO NOT__ AFFECT THE PROGRAM IN THE CONFIG FILE.
"""
defaults = {
    # Variables which affect how the simulation is run but have nothing to do with the underlying physics.
    "meta" : {
        "override_hashes"     : False,            # Override the mesh and config hashes stored in the database.
        "max_batch_size_init" : 4096,             # A starting point so that the algorithm needs less time to calibrate how much memory it can use. If this is too large the program may just crash.
        "iters"               : 4,                # How many iterations of the function the optimiser should use. Larger values take longer but may be useful for benchmarking.
        "max_scans"           : 1,                # The maximum number of times to scan over the coordinates.
        "max_scatters"        : 255,              # The maximum number of times rays can scatter off the surface. Since this is stored as an 8 bit integer it must be between 0 and 255 (inclusive). For most purposes, 255 is effectively unlimited.
        "coordinates"         : cart_scan_single, # Scanning coordinates to be used.
        "source_angle"        : np.radians(0),    # The maximum angle rays will be traced at. Think of it like a conic window function.
        # The configuration of the solver which tries to find the optimum values given the specified template.
        "solver" : {
            # The template the program uses to solve for the settings necessary to get a particular result.
            # In this case, we are only interested in the ratio of specular to diffuse
            #"template" : template_simple_scattering,
            # In this case, we are interested in the ratios of the scattering distributions as well as the reciprocal lattice vectors.
            "template" : None,
            # The properties of the image we will be scanning relative to this program's xy intensity output. We will assume the y axis has __not__ been flipped.
            "image" : {
                "x" : x,
                "y" : y,
                "flip" : False,
            },
        },
    },
    # Parameters which describe the sample and its properties.
    "sample" : {
        "shift" : np.array([0, 0, 0], dtype=np.float32), # Shift the sample by some amount
        # TODO: Add the ability to rotate the sample. Rotations don't generally commute and the sample may not be centered at the origin so this could be tricky to do intuitively...
        # A dictionary describing the function which determines the surface properties
        "properties" : {
            "function" : None, # This function dictates that all faces on the surface have identical properties.
            # These are the parameters passed to the function above.
            # In this case, the reciprocal lattice vectors are defined component by component so that we can use the solver to optimise them.
            # It's a bit verbose but it works just as well
            "kwargs"   : None
        },
    },
    # Parameters which affect the scattering
    "scattering" : {
        "function" : None, # The function which describes how the rays scatter off the surface.
    },
    # Parameters which affect how rays are detected. Pretty barebones for now. It would be nice if we could specify the detector shape, too...
    "detector" : {
        "type"     : "circle",
        "location" : shem.geometry.polar2cart([1.0, 0.0, 45.0], radians=False),  # The location of the center of the detector relative to the origin.
        "normal"   : -shem.geometry.polar2cart([1.0, 0.0, 45.0], radians=False), # The surface normal of the detector.
        "radius"   : None,                                                       # The radius of the detector aperture.
    },
    # Parameters which describe the source
    "source" : {
        "location" : shem.geometry.polar2cart([1.0, 180.0, 45.0], radians=False), # The location of the point source.
        "function" : None,                                                        # The function which decribes the distribution of rays from the point source.
    },
    # A dictionary describing how to output the data
    "display" : None,

}

##########################
# Configuration Settings #
##########################
"""
ATTENTION:
The following dictionary is the only thing that the program uses to run the simulations.
If it isn't in there, it isn't getting used.
THESE AFFECT THE PROGRAM IN THE CONFIG FILE.
"""


settings = {
    # Variables which affect how the simulation is run but have nothing to do with the underlying physics.
    "meta" : {
        "override_hashes"     : True,          # Override the mesh and config hashes stored in the database.
        "max_batch_size_init" : 1024,          # A starting point so that the algorithm needs less time to calibrate how much memory it can use. If this is too large the program may just crash.
        "max_scans"           : 2,             # The maximum number of times to scan over the coordinates.
        "max_scatters"        : 8,             # The maximum number of times rays can scatter off the surface. Since this is stored as an 8 bit integer it must be between 0 and 255 (inclusive). For most purposes, 255 is effectively unlimited.
        "coordinates"         : coordinates,   # Scanning coordinates to be used.
        "source_angle"        : np.radians(0), # The maximum angle rays will be traced at. Think of it like a conic window function.
        # The configuration of the solver which tries to find the optimum values given the specified template.
        "solver" : {
            # The template the program uses to solve for the settings necessary to get a particular result.
            # In this case, we are only interested in the ratio of specular to diffuse
            "template" : template_simple_scattering,
            # In this case, we are interested in the ratios of the scattering distributions as well as the reciprocal lattice vectors.
            #"template" : template_reciprocal_lattice,
            # The properties of the image we will be scanning relative to this program's xy intensity output. We will assume the y axis has been flipped.
            "image" : {
                "x" : x,
                "y" : y,
                "flip" : True,
            },
        },
    },
    # Parameters which describe the sample and its properties.
    "sample" : {
        # A dictionary describing the function which determines the surface properties
        "properties" : {
            "function" : surface_properties_uniform, # This function dictates that all faces on the surface have identical properties.
            # These are the parameters passed to the function above.
            # In this case, the reciprocal lattice vectors are defined component by component so that we can use the solver to optimise them.
            # It's a bit verbose but it works just as well
            "kwargs"   : {
                "properties" : {
                    "b1_x" : 1,
                    "b1_y" : 0,
                    "b1_z" : 0,
                    "b2_x" : 0,
                    "b2_y" : 1,
                    "b2_z" : 0,
                },
            },
        },
    },
    # Parameters which affect the scattering
    "scattering" : {
        "function" : scattering_function, # The function which describes how the rays scatter off the surface.
    },
    # Parameters which affect how rays are detected. Pretty barebones for now. It would be nice if we could specify the detector shape, too...
    "detector" : {
        "radius"   : 0.01,                                                       # The radius of the detector aperture.
    },
    # Parameters which describe the source
    "source" : {
        "function" : source_function,                                             # The function which decribes the distribution of rays from the point source.
    },
    # A dictionary describing how to output the data
    "display" : display,
}

