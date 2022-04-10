import os

import matplotlib.pyplot as plt
import numpy as np

import tqdm

import shem.database
from shem.definitions import *

####################
# Helper Functions #
####################
def get_output_file(args, title, parameters, extension='.png'):
    '''
    Get the file location for a particular output.
    '''
    return os.path.join(args.work_dir, title+"_"+hash_obj(parameters)+extension)

##################
# Output Styling #
##################
def show_image(img, title=""):    
    plt.imshow(img, cmap='gray')
    plt.show()
    return

def save_image(img, file_name=""):
    plt.imsave(file_name, img, cmap='gray')
    return

##################
# Data Retrieval #
##################

def load_intensity_image(path, settings=None):
    '''
    Load a greyscale intensity for analysis or manipulation.
    '''
    # Read the image and average over RGB channels to recover the original array
    data = plt.imread(path)[..., :3].mean(-1)

    # We want square data
    assert data.shape[0] == data.shape[1]

    # Set spec manually if we don't wish to use settings for debugging
    x, y, theta, phi = 0, 1, 2, 3
    if settings is None:
        spec = {
            "x" : x,
            "y" : y,
            "flip" : True,
        }
    else:
        spec = settings["meta"]["solver"]["image"]

    # Reverse transformations used to create the image.
    if spec["flip"]:
        if   spec["y"] is y:
            data = data[:, ::-1]
        elif spec["x"] is y:
            data = data[::-1]
    
    if spec["x"] < spec["y"]:
        data = data.T

    return data

def get_data_coordinates(settings, parameters, db, key, coordinate_indices=np.array([])):
    '''
    Get the data corresponding to a particular key and coordinate indices from the database.
    Avoid using iterators over the database e.g. iterrows if at all possible.
    It is extremely slow.
    Use NumPy functions.
    '''
    data = np.zeros(len(coordinate_indices))
    name = shem.database.get_name(parameters)

    # Classical signal

    if key == "signal":
        data = db.root.sig[name][coordinate_indices]

    # Average scattering at each coordinate index
    if key == "average number of times scattered":
        print("Calculating the average number of times each ray is scattered...")
        # Get a True/False array of values representing rows which have a valid coordinate index.
        valid = np.isin(db.root.rt[name].col("coordinate_index"), coordinate_indices)
        # Add each n_scat to the corresponding data index.
        data[db.root.rt[name].col("coordinate_index")[valid]] += db.root.rt[name].col("n_scat")[valid]
        # Divide by the number of data points to find the mean
        data /= settings["meta"]["max_scans"]
    
    # Average scattering at each coordinate index __for detected rays__
    if key == "average number of times scattered (detected)":
        print("Calculating the average number of times each (detected) ray is scattered...")
        detected = db.root.dt[name]
        # Get a True/False array of values representing rows which have a valid coordinate index.
        valid = np.isin(db.root.rt[name].col("coordinate_index")[detected], coordinate_indices)
        # Add each n_scat to the corresponding data index.
        data[db.root.rt[name].col("coordinate_index")[detected][valid]] += db.root.rt[name].col("n_scat")[detected][valid]
        # Determine how many values of n_scat contribute to each index.
        detected_indices, data_points = np.unique(db.root.rt[name].col("coordinate_index")[detected][valid], return_counts=True)
        # Normalise by the number of data points
        data[detected_indices] /= data_points

    return data

#####################
# Output Categories #
#####################
def xy_intensity(args, settings, parameters, db, title):
    '''
    Output greyscale image with two coordinate axes and some intensity given by a database field.
    '''
    # The specification for the plot with this title.
    spec_provided = settings["display"][title]

    # The default specification for xy intensity plots creates an image file with the y axis flipped.
    x, y, theta, phi = 0, 1, 2, 3
    default_spec = {
        "flip"    : True,           # Flip the y axis. This is necessary if we want to see the image we expect for the regular xy plot (y axis not inverted). See the camera obscura.
        "figure"  : False,          # Output a plot rather than an image. Useful if the images will be used for more plots. Defaults to False.
        "indices" : np.array([]),   # Which subset of the indices we have simulated we will be plotting.
        "x"       :  x,             # The x coordinate. Despite the name, an "xy intensity" plot can track any coordinate index
        "x_name"  : None,           # The x axis title. We can use this to e.g. include units
        "y"       :  y,             # Identical to the x axis
        "y_name"  : None,           # Identical to the x axis
        "z"       : None,           # We specify which field from the many tables we wish to plot. Here, we will just look at the signal but we could look at other properties which vary with the coordinate index like n_scat.
    }

    # Generate the specification forthis plot, applying the default values.
    spec = {k : v if k not in spec_provided.keys() else spec_provided[k] for k, v in default_spec.items()}

    # Define the output file as the figure title + settings hash
    output_file = get_output_file(args, title, parameters)

    # Get the data from the array/table in the database matching the given key.
    data = get_data_coordinates(settings, parameters, db, spec["z"], coordinate_indices=spec["indices"])
    assert data is not None

    # Reshape the data to be N x N
    N = np.sqrt(len(data))
    assert N.is_integer()
    N = int(N)
    data = data.reshape(N, N)

    # The coordinate indices are stored in the order x, y, theta, phi.
    # While we are not explicitly unravelling the data, we need to bear this order in mind.
    if spec["x"] < spec["y"]:
        data = data.T

    # Flip the y axis
    if spec["flip"]:
        if   spec["y"] is y:
            data = data[:, ::-1]
        elif spec["x"] is y:
            data = data[::-1]

    # Create either a raw image or a proper figure.
    if spec["figure"]:
        pass
    else:
        plt.imsave(output_file, data, cmap='gray')

    return

def output(args, settings, parameters):
    '''
    Using the settings provided, produced an output from the database.
    The main entry in settings used is the "display" dictionary which specifies the output format.
    '''

    # A dictionary mapping type strings to the corresponding functions.
    display_functions = {
            "xy intensity" : xy_intensity, # Create a greyscale plot in two coordinate variables of some result
    }

    # Open the database read only.
    db = shem.database.open_database_file(args)

    for title, spec in settings["display"].items():
        display_functions[spec["type"]](args, settings, parameters, db, title)

    db.close()

    return
