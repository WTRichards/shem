import numpy as np
import cupy as cp

import math
import time

import shem.configuration
import shem.simulation


############################
# Performance Optimisation #
############################

def calc_max_batch_size_rt(
        args,                 # Command line arguments
        settings,             # Simulation settings
        surface,              # Surface object
        r = 1.01,             # Multiplicative increase to be applied to rays_scan at each iteration.
        final_reduction=1/32, # The loop stops when it can't allocate the necessary memory. Reduce this final value by this fraction.
        ):

    '''
    Determine the maximum number of rays which can be traced at once.
    '''
    max_batch_size_guess = settings["meta"]["max_batch_size_init"]
    iters                = settings["meta"][              "iters"]
    max_rays             = settings["meta"][          "max_scans"] * settings["meta"]["coordinates"].shape[1]

    # Detect when an exception is raised due to a lack of memory.
    try:
        # Loop until we use a max_batch_size_guess which uses up all the GPU memory.
        while True:
            # Check if the max_batch_size_rt is greater than the size of the data set (it usually isn't), but we should check and match them
            if max_batch_size_guess > max_rays:
                max_batch_size_guess = max_rays
                break
            start_time = time.time()
            # Run the ray tracing code with the guess for max_batch_size for iters iterations.
            shem.simulation.trace_rays(args, settings, surface, max_batch_size = max_batch_size_guess, iters = iters)
            end_time = time.time()
            # Determine how long it took to run the trace_rays function. Useful for debugging output.
            time_elapsed = end_time - start_time
            if not args.quiet:
                print("max_batch_size_guess = {:,}, time elapsed per ray = {:} ns".format(max_batch_size_guess, 10**9 * time_elapsed/(iters*max_batch_size_guess)))

            # Increase the maximum number of rays geometrically
            max_batch_size_guess *= r
            max_batch_size_guess = math.floor(max_batch_size_guess)
    except:
        # Reduce the value of max_batch_size which crashes by a the final_reduction provided.
        max_batch_size_guess *= (1.0 - final_reduction)
        max_batch_size_guess = math.floor(max_batch_size_guess)
    
    if not args.quiet:
        print("Optimum max_batch_size = {:,}".format(max_batch_size_guess))
    
    # Reset RNG since the number of times we will run rays_trace is not deterministic
    np.random.seed(0)
    cp.random.seed(0)

    return max_batch_size_guess

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
    if args.use_gpu:
        if not args.quiet:
            print("Determining the optimium batch sizes on this platform...\n")
        max_batch_size_rt  = calc_max_batch_size_rt(args, settings, surface)
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
            print("Using the CPU. Setting the default batch sizes to that max_batch_size_init.\n")

    return max_batch_size_rt, max_batch_size_dt, max_batch_size_sig


###########################
# Parametric Optimisation #
###########################

def random_search(settings):
    '''
    Optimise the parameters with a random search in the vector space.
    '''
    # Get the bounds of each parameter
    bounds = shem.configuration.get_parameters_bounds(settings)
    # Get the parameters as a list
    parameters = shem.configuration.get_parameters(settings)
    # Randomise the settings
    settings = shem.configuration.randomise_settings_in_bounds(settings)
    # Ensure that the settings are in the bounds specified.
    assert shem.configuration.check_settings_in_bounds(settings) is True

    return

def gradient_descent(settings):
    '''
    Optimise the parameters using some kind of stochastic gradient descent.
    '''
    # Get the bounds of each parameter
    bounds = shem.configuration.get_parameters_bounds(settings)
    # Get the parameters as a list
    parameters = shem.configuration.get_parameters(settings)
    # Randomise the settings
    settings = shem.configuration.randomise_settings_in_bounds(settings)
    # Ensure that the settings are in the bounds specified.
    assert shem.configuration.check_settings_in_bounds(settings) is True
    
    return

def run_analysis(args):
    '''
    Analyse a provided image to determine the parameters used to create it.
    '''
    return
