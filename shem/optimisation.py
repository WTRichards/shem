import os

import numpy as np
import cupy as cp
import scipy.optimize


import math
import time

import shem.configuration
import shem.simulation

from shem.definitions import *

############################
# Performance Optimisation #
############################

def calc_max_batch_size_rt(
        args,                 # Command line arguments
        settings,             # Simulation settings
        surface,              # Surface object
        n_batches,            # Number of test batches
        r = 1.01,             # Multiplicative increase to be applied to rays_scan at each iteration.
        final_reduction=1/32, # The loop stops when it can't allocate the necessary memory. Reduce this final value by this fraction.
        ):

    '''
    Determine the maximum number of rays which can be traced at once.
    '''
    max_batch_size_guess = settings["meta"]["max_batch_size_init"]
    # Calculate the maximum number of rays it is possible to trace in a single batch.
    max_rays             = settings["meta"]["max_scans"] * settings["meta"]["coordinates"].shape[1]

    # Detect when an exception is raised due to a lack of memory.
    try:
        # Loop until we use a max_batch_size_guess which uses up all the GPU memory.
        while True:
            # Check if the max_batch_size_rt is greater than the size of the data set (it usually isn't), but we should check and match them
            if max_batch_size_guess > max_rays:
                max_batch_size_guess = max_rays
                break
            start_time = time.time()
            # Run the ray tracing code with the guess for max_batch_size for n_batches.
            shem.simulation.trace_rays(args, settings, surface, max_batch_size = max_batch_size_guess, n_batches = n_batches)
            end_time = time.time()
            # Determine how long it took to run the trace_rays function. Useful for debugging output.
            time_elapsed = end_time - start_time
            if not args.quiet:
                print("max_batch_size_guess = {:,}, time elapsed per ray = {:} ns...".format(max_batch_size_guess, 10**9 * time_elapsed/(n_batches*max_batch_size_guess)))

            # Increase the maximum number of rays geometrically
            max_batch_size_guess *= r
            max_batch_size_guess = math.floor(max_batch_size_guess)
    except:
        # Do nothing. We want the CuPy to run out of memory so we know we can use the batch size just below that one.
        pass
    
    # Reduce the value of max_batch_size which crashes by a the final_reduction provided.
    max_batch_size_guess *= (1.0 - final_reduction)
    max_batch_size = math.floor(max_batch_size_guess)
    
    if not args.quiet:
        print("Optimum max_batch_size = {:,}...".format(max_batch_size))
    
    return max_batch_size

def calc_max_batch_sizes(
        args,                 # Command line arguments
        settings,             # Simulation settings
        surface,              # Surface object
        r = 1.01,             # Multiplicative increase to be applied to rays_scan at each iteration.
        final_reduction=1/50, # The loop stops when it can't allocate the necessary memory. Reduce this final value by this fraction.
        ):
    '''
    A wrapper around the three functions which perform ray batch memory usage optimisation for the rt, dt and sig steps.
    '''
    # How many batches to use per optimisation
    n_batches = settings["meta"]["optimisation_batches"]
    # TODO: Allow the user to set r and final reduction.

    if args.use_gpu:
        if not args.quiet:
            print("Determining the optimium batch sizes for ray tracing, ray detection and signal convolution on this platform...")
        max_batch_size_rt  = calc_max_batch_size_rt(args, settings, surface, n_batches)
        # TODO: Implement optimisers for the dt and sig steps. For now we will just use the value for the rt step, which is smaller than it could be for these.
        max_batch_size_dt  = max_batch_size_rt
        max_batch_size_sig = max_batch_size_rt
    else:
        if not args.quiet:
            print("Using max_batch_size_init as the default batch size on the CPU...")
        max_batch_size_rt  = settings["meta"]["max_batch_size_init"]
        max_batch_size_dt  = settings["meta"]["max_batch_size_init"]
        max_batch_size_sig = settings["meta"]["max_batch_size_init"]
        if not args.quiet:
            print("Using the CPU. Setting the default batch sizes to that max_batch_size_init...")

    return max_batch_size_rt, max_batch_size_dt, max_batch_size_sig


###########################
# Parametric Optimisation #
###########################

def calc_chi2(image, test, processing=lambda data: data):
    '''
    Calculate the sum of the squared differences between the image and the result, then divide by the expected value.
    '''
    return np.nan_to_num( (processing(image) - processing(test))**2 / processing(test), nan=0, posinf=0, neginf=0 ).sum()

def calc_mse(image, test, processing=lambda data: data):
    '''
    Calculate the sum of the squared differences between the image and the result, then divide by the expected value.
    '''
    return np.nan_to_num( (processing(image) - processing(test))**2, nan=0, posinf=0, neginf=0 ).sum()

def run_analysis(args):
    '''
    Analyse a provided image to determine the parameters used to create it.
    '''
    
    # Load the settings
    settings = shem.configuration.get_settings(args)

    # Load the image using the settings for the solver.
    image = shem.display.load_intensity_image(os.path.join(args.work_dir, args.image), settings)
    image /= image.max()

    # Get the relevant coordinate indices
    indices = settings["meta"]["solver"]["indices"]
    # The threshold scales with the image size
    threshold = settings["meta"]["solver"]["threshold"]
    
    # Load the surface object into memory
    surface = shem.surface.load_surface(args, settings)

    # Calculate the maximum number of rays we can process on the GPU for each step.
    max_batch_sizes = shem.optimisation.calc_max_batch_sizes(args, settings, surface)
     
    # Open the database file.
    db = shem.database.open_database_file(args, mode="r+")
    # Parameters table in the database
    params_table = db.root.metadata.params

    # Specify the bounds - we could also specify constraints here...
    bounds = shem.configuration.get_parameters_bounds(settings)

    def simulation_wrapper(parameters):
        settings = shem.configuration.get_settings(args)

        # Loop parameters back into the allowed region
        for i, parameter in enumerate(parameters):
            if parameter < bounds[i][0] or parameter > bounds[i][1]:
                parameters[i] = (parameter - bounds[i][0]) % (bounds[i][1] - bounds[i][0]) + bounds[i][0]

        # Apply the new parameters to the settings
        settings = shem.configuration.set_setting_values(settings, parameters)
        #shem.configuration.get_parameters(settings)

        # Ensure that the settings are in the bounds specified.
        assert shem.configuration.check_settings_in_bounds(settings) is True
        
        # Create the necessary tables and arrays
        db_tuple = shem.database.create_new_simulation_db(db, settings, parameters)

        # Run the simulation
        shem.simulation.run_simulation(args, settings, db_tuple, surface, max_batch_sizes)

        # Get the result from the database
        simulation_result = np.reshape(db_tuple[2][indices], image.shape)
        simulation_result = simulation_result / simulation_result.max()

        # Calculate the Chi2 value for denoised images
        chi2 = calc_chi2(image, simulation_result) / image.size

        # Print the parameters used and the chi2 value.
        if not args.quiet:
            print("Chi2 value = " + str(chi2) + "\nParameters:\n" + str(parameters))

        # Append the results to the database
        p = params_table.row
        p['hashed'] = hash_obj(parameters)
        p['values'] = parameters
        p['chi2']   = chi2
        p.append()
        params_table.flush()

        return chi2

    # Start in the center of the bounds.
    x0 = np.array([ (bound[0] + bound[1])/2 for bound in bounds ])
    # Override for now.
    #x0 = np.array([0.9, 0.1, 0.95])

    # Seed the RNG
    np.random.seed(0)
    cp.random.seed(0)

    # Minimise the chi2 between the result and image, subject to the bounds specified, starting in the middle of them with a particular tolerance for termination.
    # See SciPy documentation for more details.
    result = scipy.optimize.basinhopping(simulation_wrapper, x0)

    # Output the result of the minimisation
    print(result)

    # Close the database file
    db.close()

    return
