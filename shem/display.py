import os

import matplotlib.pyplot as plt
import numpy as np
# Set the print threshold to for stdout output
np.set_printoptions(threshold=32*32)

import cv2

import tqdm

import shem.database
from shem.definitions import *

####################
# Helper Functions #
####################
def get_output_file(args, title, parameters, extension='.png'):
    '''
    Get the file location for a particular output.
    This will be {config hash}/{title}_{parameters_hash}.{ext}
    '''
    return os.path.join(args.work_dir, hash_file(os.path.join(args.work_dir, args.config)), title+"_"+hash_obj(parameters)+extension)

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
    x, y, z, theta, phi = [i for i in range(5)]
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
        if   spec["x"] == y:
            data = data[:, ::-1]
        elif spec["y"] == y:
            data = data[::-1]

    return data

def get_data_coordinates(settings, parameters, db, key, coordinate_indices=np.array([])):
    '''
    Get the data corresponding to a particular key and coordinate indices from the database.
    Avoid using iterators over the database e.g. iterrows if at all possible.
    It is extremely slow.
    Use NumPy functions.
    '''
    # The name of the relevant databases / arrays
    name = shem.database.get_name(parameters)

    # Get settings
    coordinates = settings["meta"]["coordinates"]
    max_scans = settings["meta"]["max_scans"]
    
    # Sum over all coordinates and slice at the end. This is less efficient, but easier.
    data = np.zeros(coordinates.shape[1])

    # Classical signal

    if key == "signal":
        data = db.root.sig[name]

    # Average scattering at each coordinate index
    if key == "average number of times scattered":
        # Add at each coordinate index.
        np.add.at(data, db.root.rt[name].col("coordinate_index"), db.root.rt[name].col("n_scat"))
        # Normalise by the number of scans
        data /= max_scans
    
    # Average scattering at each coordinate index __for detected rays__
    if key == "average number of times scattered (detected)":
        # Find the detected rays
        detected = db.root.dt[name]
        # Add at only the coordinate indices detected.
        np.add.at(data, db.root.rt[name].col("coordinate_index")[detected], db.root.rt[name].col("n_scat")[detected])
        # Count how many rays are detected at each index
        unique_indices, index_count = np.unique(db.root.rt[name].col("coordinate_index")[detected], return_counts=True)
        # Normalise each data point
        data[unique_indices] /= index_count

    return data[coordinate_indices]

def get_data_count(settings, parameters, db, key, coordinate_indices=np.array([])):
    '''
    Get the data corresponding to a particular key and coordinate indices from the database.
    Output the unique elements of that data, the number of such elements and an error, if possible.
    '''
    # The name of the relevant databases / arrays
    name = shem.database.get_name(parameters)

    # Get settings
    coordinates = settings["meta"]["coordinates"]
    max_scans   = settings["meta"]["max_scans"]
    
    # Get table indices satisfying the criteria.
    indices = np.isin(db.root.rt[name].col("coordinate_index"), coordinate_indices)

    # Classical signal
    if key == "signal":
        data = db.root.sig[name][coordinate_indices]

    # Average scattering at each coordinate index
    if key == "number of times scattered":
        data = db.root.rt[name].col("n_scat")[indices]
    
    # Average scattering at each coordinate index __for detected rays__
    if key == "number of times scattered (detected)":
        # Find the detected rays
        detected = db.root.dt[name]
        data = db.root.rt[name].col("n_scat")[indices][detected]

    return np.unique(data, return_counts=True)

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
    x, y, z, theta, phi = [i for i in range(5)]
    default_spec = {
        "flip"    : True,           # Flip the y axis. This is necessary if we want to see the image we expect for the regular xy plot (y axis not inverted). See the camera obscura.
        "FT"      : False,          # Fourier transform the image.
        "denoise" : False,          # Apply a denoising filter
        "figure"  : False,          # Output a plot rather than an image. Useful if the images will be used for more plots. Defaults to False.
        "stdout"  : False,          # Output a the values to stdout scaled to 0, 1 or 2 and scaled down to a 32 x 32 grid. Defaults to False.
        "indices" : np.array([]),   # Which subset of the indices we have simulated we will be plotting.
        "x"       :  x,             # The x coordinate. Despite the name, an "xy intensity" plot can track any coordinate index
        "x_name"  : None,           # The x axis title. We can use this to e.g. include units
        "y"       :  y,             # Identical to the x axis
        "y_name"  : None,           # Identical to the x axis
        "z"       : None,           # We specify which field from the many tables we wish to plot. Here, we will just look at the signal but we could look at other properties which vary with the coordinate index like n_scat.
    }


    # Generate the specification for this plot, applying the default values.
    spec = {k : v if k not in spec_provided.keys() else spec_provided[k] for k, v in default_spec.items()}
    
    # Get the indices and raw coordinates
    indices = spec["indices"]
    coordinates = settings["meta"]["coordinates"]

    # Define the output file as the figure title + settings hash
    output_file = get_output_file(args, title, parameters)

    # Get the data from the array/table in the database matching the given key.
    data = get_data_coordinates(settings, parameters, db, spec["z"], coordinate_indices=indices)
    assert data is not None

    # Reshape the data to be N x N
    N = np.sqrt(len(data))
    assert N.is_integer()
    N = int(N)
    data = data.reshape(N, N)

    # Transpose the data if necessary since reshaping does not respect the original coordinates
    if spec["x"] < spec["y"]:
        data = data.T

    """
    # We need to transpose the indices for the data to make sense if we are not scanning in x and y.
    if (spec["x"] == x and spec["y"] == y) or (spec["x"] == y and spec["y"] == x):
        pass
    else:
        data = data.T
    """

    # Flip the y axis
    if spec["flip"]:
        if   spec["x"] == y:
            data = data[:, ::-1]
        elif spec["y"] == y:
            data = data[::-1]

    # Apply a denoising filter
    if spec["denoise"]:
        data = denoise(data)

    # Fourier transform the data and shift so that 0 frequency components are in the center.
    if spec["FT"]:
        data = np.abs(np.fft.fftshift(np.fft.fft2(data)))

    # Data Output
    if spec["stdout"]:
        # Scale down the image/array output
        if N > 32:
            data = data.reshape(32, N//32, 32, N//32).sum((1, 3))
        # Scale the numbers in the output to between 0 and 2
        max_val = np.max(data)
        data /= max_val
        data *= 2
        print(np.array(np.rint(data), dtype=int))

    # Create either a raw image or a proper figure.
    if spec["figure"]:
        # Create the figure and axes
        fig, ax = plt.subplots()
        
        # Set the title
        ax.set_title(title)

        # Display the data in greyscale
        plt.imshow(data, cmap='gray')
       
        # Start with the first element in the bottom corner
        #ax.invert_yaxis()

        # Set the axis ticks
        x_min, x_max, y_min, y_max = coordinates[spec["x"]][indices].min(), coordinates[spec["x"]][indices].max(), coordinates[spec["y"]][indices].min(), coordinates[spec["y"]][indices].max(),
        ticks = np.array([i*(N//8) for i in range(9)])
        
        # Configure the ticks
        # Round the axis ticks
        rounded = lambda x: '{0:.3g}'.format(x)
        ax.set_xticks(ticks, list(map(rounded, (ticks/N)*(x_max-x_min)+x_min)))
        ax.set_yticks(ticks, list(map(rounded, (ticks/N)*(y_max-y_min)+y_min)))

        # Label the axes
        ax.set_xlabel(spec["x_name"])
        ax.set_ylabel(spec["y_name"])
        
        # Save the figure
        plt.savefig(output_file, dpi=600)
    else:
        # Save the image
        plt.imsave(output_file, data, cmap='gray')

    return

def bar_chart(args, settings, parameters, db, title):
    '''
    Output a histogram-like figure derived from:
    The finite elements in a database column e.g. the number of times rays scatter.
    The occurence of that quantity.
    '''
    # The specification for the plot with this title.
    spec_provided = settings["display"][title]

    # The default specification for xy intensity plots creates an image file with the y axis flipped.
    x, y, z, theta, phi = [i for i in range(5)]
    default_spec = {
        "figure"      : True,                        # Output a plot rather than an image. Useful if the images will be used for more plots. Defaults to True.
        "stdout"      : False,                       # Output a the values to stdout scaled to 0, 1 or 2 and scaled down to a 32 x 32 grid. Defaults to False.
        "logarithmic" : False,                       # Plot the bar height on a logarithmic axis.
        "indices"     : np.array([]),                # Which subset of the indices we have simulated we will be plotting.
        "x"           : None,                        # The value we are tracking.
        "x_name"      : "Number of Times Scattered", # The x axis title. We can use this to e.g. include units
    }

    # Generate the specification forthis plot, applying the default values.
    spec = {k : v if k not in spec_provided.keys() else spec_provided[k] for k, v in default_spec.items()}

    # Define the output file as the figure title + settings hash
    output_file = get_output_file(args, title, parameters)

    # Get the data from the array/table in the database matching the given key.
    data = get_data_count(settings, parameters, db, spec["x"], coordinate_indices=spec["indices"])
    assert data is not None

    # Data Output
    if spec["stdout"]:
        # We should be able to print the data
        print(data)

    # Create either a raw image or a proper figure.
    if spec["figure"]:
        fig, ax = plt.subplots()
        
        ax.set_title(title)

        ax.bar(data[0], data[1], width=1, fill=True, log=spec["logarithmic"])

        ax.set_xlabel(spec["x_name"])
        if spec["logarithmic"]:
            ax.set_ylabel("log(count)")
        else:
            ax.set_ylabel("count")
        
        plt.savefig(output_file, dpi=600)
    else:
        pass

    return

def output(args, settings, parameters):
    '''
    Using the settings provided, produced an output from the database.
    The main entry in settings used is the "display" dictionary which specifies the output format.
    '''

    # A dictionary mapping type strings to the corresponding functions.
    display_functions = {
            "xy intensity" : xy_intensity, # Create a greyscale plot in two coordinate variables of some result
            "bar chart"    : bar_chart,       # Create a bar chart using two variables. One, a finite Python list, the other some continuous value.
    }

    # Open the database read only.
    db = shem.database.open_database_file(args)

    # Create the output directory
    os.makedirs(os.path.join(args.work_dir, hash_file(os.path.join(args.work_dir, args.config))), exist_ok=True)

    # Iterate over each display specification
    tqdm_iterator = tqdm.tqdm(settings["display"].keys(), leave=True)
    for title in tqdm_iterator:
        tqdm_iterator.set_description("generating {}".format(title))
        # Limit the plots we can produce without the database.
        if args.enable_database or (settings["display"][title]["type"] == "xy intensity" and settings["display"][title]["z"] == "signal"):
            display_functions[settings["display"][title]["type"]](args, settings, parameters, db, title)
    
    db.close()

    return
