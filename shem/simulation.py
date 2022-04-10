import os, sys
import time

import trimesh

import numpy as np
import cupy  as cp

from shem.definitions import *
import shem.configuration
import shem.detection
import shem.optimisation
import shem.source
import shem.surface

#########################
# Core Simulation Steps #
#########################

# TODO: Modify this to use a for loop and be nice like the others.
def trace_rays(
        args,                  # Command line arguments
        settings,              # Simulation settings
        surface,               # Surface object
        rt_table = None,       # Table in which to save the ray tracing results
        max_batch_size = 1024, # Maximum number of rays to trace per batch.
        iters = 1,             # Number of times to run through the main loop. Useful for calibration.
        max_scatters = None,   # Override max_scatters in settings. Useful for debugging
        ):
    '''
    Trace the rays which scatter off the sample using the given settings and store the result in a database.
    '''
    
    xp = get_xp(args)

    # Get variables from settings for convenience and move them to the GPU, if necessary
    coordinates         = xp.array(settings[      "meta"][ "coordinates"])
    max_scans           =          settings[      "meta"][   "max_scans"]
    source_angle        =          settings[      "meta"]["source_angle"]
    source_location     = xp.array(settings[    "source"][    "location"])
    scattering_function =          settings["scattering"][    "function"]
    if max_scatters is None:
        max_scatters    =          settings[      "meta"]["max_scatters"]

    # Calculate the maximum number of rays we are allowed to trace.
    max_rays_traced = max_scans * coordinates.shape[1]

    # Check if this table is complete. If it is not, empty it.
    # We cannot start in the middle because the RNG resets each time this function is called.
    # TODO: Save the RNG state as an attribute in the table so we can stop and resume the simulation.
    # When we wish to continue, we can load the RNG state here.
    if rt_table is not None:
        if   rt_table.nrows == max_rays_traced:
            print("Ray tracing table: " + rt_table.name + " is complete. Continuing...")
            return
        elif rt_table.nrows != 0:
            print("Ray tracing table: " + rt_table.name + " is only partially complete. Purging the table...")
            # Delete all rows
            rt_table.remove_rows(0)
    
    # Seed the random data. We do this every time this function is called to ensure consistency every time this program is run.
    np.random.seed(0)
    xp.random.seed(0)

    # Track how many rays we have traced.
    rays_traced = rt_table.nrows


    # Allocate memory for the initial rays.
    # We do this once at the start for performance.
    rays_i = xp.empty((2, max_batch_size, 3), dtype=xp.float32)

    # Track when we start.
    start_time = time.time()

    ####################
    # Useful Functions #
    ####################

    ##################################################################################################################

    # We will use a function to save the ray tracing data to the table to avoid reusing code for different batch sizes.
    def save_results(args, batch_size, coordinate_indices, a_polar_i, rays_f, n_scat):
        '''
        Save the results of a batch.
        '''
        # Calculate the properties of the final rays
        a_polar_f = shem.geometry.cart2polar(rays_f[0])[..., 1:]
        b_f       =                          rays_f[1]
        # Move the data off the GPU
        if args.use_gpu:
            a_polar_i          =          a_polar_i.get()
            a_polar_f          =          a_polar_f.get()
            b_f                =                b_f.get()
            n_scat             =             n_scat.get()
            coordinate_indices = coordinate_indices.get()

        # TODO: Try and look for a better way to do this.
        # This is not a bottleneck now since it takes longer to calculate the data than write it but this can probably be parallelised.
        for i in range(batch_size):
            ray = rt_table.row
            ray['coordinate_index'] = coordinate_indices[i]
            ray['a_polar_i']        =          a_polar_i[i]
            ray['a_polar_f']        =          a_polar_f[i]
            ray['b_f']              =                b_f[i]
            ray['n_scat']           =             n_scat[i]
            ray.append()

        rt_table.flush()
        return

    def set_initial_rays(rays_i, coordinate_indices):
        '''
        Set the intial rays based on the indices of the coordinates of interest.
        Returns the deviation in polar coordinates of a from a 'perfect' source.
        '''
        a = rays_i[0]
        b = rays_i[1]
        # Determine the point source direction vector.
        # We will trace rays in a cone defined by source_angle and convolve with various source distributions later.
        # We also want the angular distribution relative to the source vector for analysis later, so return that too. 
        a[:], a_polar_i = shem.source.direction(a, source_location, source_angle, coordinates, coordinate_indices)
        # Determine the point source origin vector.
        b[:] = shem.source.origin(              b, source_location,               coordinates, coordinate_indices)

        return a_polar_i

    def progress_information(rays_traced):
        if args.verbose:
            print("total rays traced, {:} total seconds elapsed...".format(rays_traced, time.time()-start_time), end='\r')
        return


    ##################################################################################################################


    ###################
    # Iterations Loop #
    ###################

    # TODO: Create a table which is used for testing and benchmarking
    # Run for a certain number of iterations, looping over the rays then exit.
    if rt_table is None:
        for i in range(iters):
            # Determine the coordinate indices of the rays we will be tracing.
            coordinate_indices = xp.arange(rays_traced, rays_traced+max_batch_size) % coordinates.shape[1]

            # Set the initial values of the rays and get the polar coordinate displacements from a perfect source.
            a_polar_i = set_initial_rays(rays_i, coordinate_indices)

            # Scatter the rays off the surface
            rays_f, n_scat = shem.scattering.scatter(rays_i, surface, scattering_function, max_scatters)

            # Keep track of the number of rays we have traced.
            rays_traced += max_batch_size
            progress_information(rays_traced)
        return


    #############
    # Scan Loop #
    #############

    # Calculate how many batches will we need to run through the entire table.
    n_batches = max_rays_traced // max_batch_size
    # Calculate the size of the final batch.
    last_batch_size = max_rays_traced % max_batch_size

    print(n_batches, last_batch_size)

    # Iterate over every batch
    for batch in range(n_batches):
        # Determine the coordinate indices of the rays we will be tracing.
        coordinate_indices = xp.arange(rays_traced, rays_traced+max_batch_size) % coordinates.shape[1]
        
        # Set the initial values of the rays and get the polar coordinate displacements from a perfect source.
        a_polar_i = set_initial_rays(rays_i, coordinate_indices)

        # Scatter the rays off the surface
        rays_f, n_scat = shem.scattering.scatter(rays_i, surface, scattering_function, max_scatters)

        # Save our results to the database.
        save_results(args, max_batch_size, coordinate_indices, a_polar_i, rays_f, n_scat)
        
        # Keep track of the number of rays we have traced.
        rays_traced += max_batch_size
        print(rays_traced)
        progress_information(rays_traced)


    # If necessary, run the final batch with a different batch size
    if last_batch_size != 0:
        # Reallocate memory for rays_i for a smaller batch. This is not strictly necessary but is generally sensible.
        rays_i = xp.empty((2, last_batch_size, 3), dtype=xp.float32)

        # Determine the coordinate indices of the rays we will be tracing.
        coordinate_indices = xp.arange(rays_traced, rays_traced+last_batch_size) % coordinates.shape[1]
        
        # Set the initial values of the rays and get the polar coordinate displacements from a perfect source.
        a_polar_i = set_initial_rays(rays_i, coordinate_indices)

        # Scatter the rays off the surface
        rays_f, n_scat = shem.scattering.scatter(rays_i, surface, scattering_function, max_scatters)

        # Save our results to the database.
        save_results(args, last_batch_size, coordinate_indices, a_polar_i, rays_f, n_scat)
        
        # Keep track of the number of rays we have traced.
        rays_traced = max_rays_traced
        progress_information(rays_traced)


    # Inform us about the success conditions
    if args.verbose:
        if rt_table is None:
            print("Successfully completed " + str(iters) + " iterations...")
        else:
            print("Successfully completed " + str(max_scans) + " scans...")

    # Return the resultant rays of the scattering. Useful for debugging
    return rays_f

def detect_rays(
        args,          # Command line arguments
        settings,      # Simulation settings
        rt_table,      # Table from which to read the ray tracing results
        dt_arr = None, # Array in which to save the detected rays
        max_rays_batch = 1024, # Maximum number of rays to detect per batch.
        ):
    '''
    Detect the rays within the ray tracing database.
    '''
    # Seed the random data.
    np.random.seed(0)
    cp.random.seed(0)
    xp = get_xp(args)

    # We cannot detect more rays than the length of the corresponding table.
    max_rays_detected = len(rt_table)

    # How many batches will we need to run through the entire table
    n_batches = (max_rays_detected // max_rays_batch) + 1
    for batch in range(n_batches):
        # Indices of interest
        batch_range = [batch*max_rays_batch, (batch+1)*max_rays_batch]
        
        # If the upper index is longer than the table, reduce it.
        if batch_range[1] > max_rays_detected:
            batch_range[1] = max_rays_detected

        # Get the rays from the table and convert back to Cartesians.
        rays = xp.stack((
            shem.geometry.unit_vector(xp.array(rt_table.read(*batch_range, field="a_polar_f"))),
            xp.array(rt_table.read(*batch_range, field="b_f"))
        ))

        # Get the coordinate indices for each ray
        coordinate_indices = xp.array(rt_table.read(*batch_range, field="coordinate_index"))
        
        # Determine which rays are detected. We need to add on the starting index of the batch.
        is_detected = shem.detection.detect(rays, settings, coordinate_indices)
        
        # Load the result from the GPU
        if args.use_gpu:
            is_detected = is_detected.get()

        # Append the indices we have detected.
        dt_arr[batch_range[0]:batch_range[1]] = is_detected

        # Save the changes
        dt_arr.flush()

    dt_arr.attrs.complete = True
    dt_arr.flush()
    return

def source_convolve_rays(
        args,           # Command line arguments
        settings,       # Simulation settings
        rt_table,       # Table from which to read the ray tracing results
        dt_arr,         # Array from which to read the detected rays
        sig_arr = None, # Array in which to save the signal at each coordinate
        max_rays_batch = 1024, # Maximum number of rays to convolve and sum per batch.
        ):
    '''
    Calculate the signal experienced at each coordinate due to the rays detected.
    '''
    # Seed the random data.
    np.random.seed(0)
    cp.random.seed(0)
    xp = get_xp(args)

    source_function = settings["source"]["function"]
    source_angle    = settings[  "meta"]["source_angle"]
   
    # Determine the indices of the rays detected.
    detected_rays = np.where(dt_arr)

    # Check if we have already completed the signal calculation procedure on this table.
    if sig_arr.attrs.complete:
        print("This signal calculation procedure has already been completed. Continuing...")
        return
    else:
        max_rays_signal = len(detected_rays)
        # Create the array which will store the signal we compute.
        signal = xp.zeros(len(sig_arr), dtype=xp.float32)

    # How many batches will we need to run through the entire table
    n_batches = (max_rays_signal // max_rays_batch) + 1
    for batch in range(n_batches):
        # Indices of interest
        batch_range = [batch*max_rays_batch, (batch+1)*max_rays_batch]
        
        # If the upper index is longer than the table, reduce it.
        if batch_range[1] > max_rays_signal:
            batch_range[1] = max_rays_signal

        # Get the indices of the detected rays
        detected_ray_coordinate_indices = detected_rays[batch_range[0]:batch_range[1]]
        # Get the coordinate indices of each detected ray
        coordinate_indices = rt_table.read_coordinates(detected_ray_coordinate_indices, field="coordinate_index")
        # Get the source ray displacement relative to a 'perfect' ray in spherical polars with the perfect ray as the z axis.
        theta, phi = xp.array(rt_table.read_coordinates(detected_ray_coordinate_indices, field="a_polar_i")).T
        
        # Calculate the signal each ray in this batch will produce. If source_angle is 0 there is no point. Each ray detected has the same weight.
        if source_angle == 0:
            batch_signal = xp.ones_like(theta)
        else:
            batch_signal = shem.source.calc_source_function(theta, phi, settings)

        # Add the batch signal to the total signal
        signal[detected_ray_coordinate_indices] += batch_signal

    # Load the result from the GPU
    if args.use_gpu:
        signal = signal.get()

    # Save the signal we have detected to the array.
    sig_arr[:] = signal

    # The array is complete
    sig_arr.attrs.complete = True
    sig_arr.flush()
    return


#######################
# Simulation Wrappers #
#######################

def run_simulation(args, settings, db_tuple, surface, max_batch_sizes):
    '''
    Run the simulation with the given arguments, settings, database, surface and batch sizes.
    '''
    # Unpack the tuple containing the relevant arrays and tables
    rt_table, dt_arr, sig_arr = db_tuple
    
    # Unpack the tuple containing the maximum number of rays per batch.
    max_batch_size_rt, max_batch_size_dt, max_batch_size_sig = max_batch_sizes
    
    # Apply any modifications to the surface
    surface = shem.surface.modify_surface(surface, settings)

    # Trace and scatter the rays off the surface
    trace_rays(args, settings, surface, rt_table, max_batch_size_rt)

    # Detect the rays which were scattered.
    detect_rays(args, settings, rt_table, dt_arr, max_batch_size_dt)

    # Calculate the signal from each of the detected rays.
    source_convolve_rays(args, settings, rt_table, dt_arr, sig_arr, max_batch_size_sig)
    
    return

def run_simulation_default(args):
    '''
    Run the default simulation using settings.
    '''
    # No parameters supplied.
    parameters = None

    # Get the settings from the config file
    settings = shem.configuration.get_settings(args)
    
    # Load the surface object into memory
    surface = shem.surface.load_surface(args, settings)

    # Calculate the maximum number of rays we can process on the GPU for each step.
    max_batch_sizes = shem.optimisation.calc_max_batch_sizes(args, settings, surface)

    # Open the database file.
    db = shem.database.open_database_file(args, mode="r+")

    # Create the necessary tables and arrays
    db_tuple = shem.database.create_new_simulation_db(db, settings, parameters)
    
    # Run the simulation
    run_simulation(args, settings, db_tuple, surface, max_batch_sizes)

    # Close the database. The output function will reopen it in read-only mode.
    db.close()
    
    # Output the simulation result using the specification in settings.
    shem.display.output(args, settings, parameters)

    return

#############
# Debugging #
#############

def trace_single_ray(args):
    '''
    Trace a single ray and display the result in a 3D interactive graphic.
    Useful for debugging
    '''
    
    # No parameters supplied.
    parameters = None

    # Get the settings from the config file
    settings = shem.configuration.get_settings(args)
    
    # Load the surface object into memory
    surface = shem.surface.load_surface(args, settings)

    # Apply any modifications to the surface
    surface = shem.surface.modify_surface(surface, settings)

    # We only trace a single ray
    max_batch_size_rt = 1
    
    # Trace
    trace_rays(args, settings, surface, rt_table, max_batch_size_rt)

    return
