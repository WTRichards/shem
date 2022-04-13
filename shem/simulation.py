import os, sys
import time

import trimesh
import tqdm

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
        n_batches = 8,         # Number of batches for testing
        ):
    '''
    Trace the rays which scatter off the sample using the given settings and store the result in a database.
    '''
    
    xp = get_xp(args)

    # Get variables from settings for convenience and move them to the GPU, if necessary
    coordinates         = xp.array(settings[      "meta"][ "coordinates"])
    max_scans           =          settings[      "meta"][   "max_scans"]
    source_location     = xp.array(settings[    "source"][    "location"])
    source_function     =          settings[    "source"][    "function"]
    scattering_function =          settings["scattering"][    "function"]
    max_scatters        =          settings[      "meta"]["max_scatters"]

    # Calculate the maximum number of rays we are allowed to trace.
    max_rays_traced = max_scans * coordinates.shape[1]

    # Check if this table is complete. If it is not, empty it.
    # We cannot start in the middle because the RNG resets each time this function is called.
    # TODO: Save the RNG state as an attribute in the table so we can stop and resume the simulation.
    # When we wish to continue, we can load the RNG state here.
    if rt_table is not None:
        name = rt_table.name
        if   rt_table.nrows == max_rays_traced:
            print("Ray tracing table: " + rt_table.name + " is complete. Continuing...")
            return
        elif rt_table.nrows != 0:
            print("Ray tracing table: " + rt_table.name + " is only partially complete. Purging the table...")
            # Delete all rows
            rt_table.remove_rows(0)

        # Track how many rays we have traced.
        rays_traced = rt_table.nrows

        # Calculate how many batches will we need to run through the entire table.
        n_batches = (max_rays_traced - rays_traced) // max_batch_size

        # Calculate the size of the final batch.
        last_batch_size = (max_rays_traced - rays_traced) % max_batch_size
    else:
        # Hardcode the batch size for optimisation.
        name = "Testing"
        rays_traced = 0
        max_rays_traced = n_batches * max_batch_size
        last_batch_size = 0


    # Seed the random data. We do this every time this function is called to ensure consistency every time this program is run.
    np.random.seed(0)
    xp.random.seed(0)

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
    def save_results(args, coordinate_indices, a_polar_i, rays_f, n_scat):
        '''
        Save the results of a batch.
        '''
        
        batch_size = coordinate_indices.size

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

    ##################################################################################################################

    #############
    # Main Loop #
    #############

    # Scatter each ray batch
    tqdm_iterator = tqdm.trange(n_batches+1, leave=True)
    for batch in tqdm_iterator:
        tqdm_iterator.set_description("{} : {:,} total rays traced, {:,} scans complete".format(name, rays_traced, rays_traced // coordinates.shape[1]))
        # Check whether we are on the final batch.
        if batch == n_batches:
            # Exit the loop if there is no final batch
            if last_batch_size == 0:
                continue
            batch_range = [rays_traced, rays_traced+last_batch_size]
            # Reallocate memory
            rays_i = xp.empty((2, last_batch_size, 3), dtype=xp.float32)
        else:
            batch_range = [rays_traced, rays_traced+max_batch_size]

        # Determine the coordinate indices of the rays we will be tracing.
        coordinate_indices = xp.arange(*batch_range) % coordinates.shape[1]
        
        # Set the initial values of the rays and get the polar coordinate displacements from a perfect source.
        #a_polar_i = set_initial_rays(rays_i, coordinate_indices)
        a_polar_i = shem.source.get_source_rays(rays_i, source_location, source_function, coordinates, coordinate_indices)

        # Scatter the rays off the surface
        rays_f, n_scat = shem.scattering.scatter(rays_i, surface, scattering_function, max_scatters)

        # Save our results to the database.
        if rt_table is not None:
            save_results(args, coordinate_indices, a_polar_i, rays_f, n_scat)
        
        # Keep track of the number of rays traced.
        rays_traced += batch_range[1] - batch_range[0]

    return

def detect_rays(
        args,          # Command line arguments
        settings,      # Simulation settings
        rt_table,      # Table from which to read the ray tracing results
        dt_arr = None, # Array in which to save the detected rays
        max_batch_size = 1024, # Maximum number of rays to detect per batch.
        ):
    '''
    Detect the rays within the ray tracing database.
    '''
    
    xp = get_xp(args)

    # We cannot detect more rays than the length of the corresponding table. This is simpler than getting the result from settings.
    max_rays_detected = rt_table.nrows
    
    # Check if this table is array is complete. If it is not, set it all to False.
    # If we wanted to have it be able to start from an arbitrary point, we would need to find the first True value.
    if dt_arr is not None:
        if   dt_arr.attrs.complete:
            print("Ray detection array: " + dt_arr.name + " is complete. Continuing...")
            return
        elif dt_arr[:].any():
            print("Ray tracing table: " + rt_table.name + " is only partially complete. Purging the table...")
            # Set all elements to False
            dt_arr[:] = False
    
    # Find the index of the last ray detected.
    rays_detected = np.where(dt_arr)[0]
    if rays_detected.size == 0:
        rays_detected = 0
    
    # How many batches will we need to run through the entire table
    n_batches = (max_rays_detected - rays_detected) // max_batch_size
    
    # Calculate the size of the final batch.
    last_batch_size = (max_rays_detected - rays_detected) % max_batch_size

    # Perform the ray detection on each batch
    tqdm_iterator = tqdm.trange(n_batches+1, desc=dt_arr.name + " : detecting rays", leave=True)
    for batch in tqdm_iterator:
        # Check whether we are on the final batch.
        if batch == n_batches:
            # Exit the loop if there is no final batch
            if last_batch_size == 0:
                break
            batch_range = [rays_detected, rays_detected+last_batch_size]
        else:
            batch_range = [rays_detected, rays_detected+ max_batch_size]
        
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
        
        # Keep track of the number of rays we have detected.
        rays_detected += batch_range[1] - batch_range[0]

    # Mark this array as completed
    dt_arr.attrs.complete = True

    return

def sum_detected_rays(
        args,           # Command line arguments
        settings,       # Simulation settings
        rt_table,       # Table from which to read the ray tracing results
        dt_arr,         # Array from which to read the detected rays
        sig_arr = None, # Array in which to save the signal at each coordinate
        max_batch_size = 1024, # Maximum number of rays to convolve and sum per batch.
        ):
    '''
    Calculate the signal experienced at each coordinate due to the rays detected.
    This does not need to be on the GPU.
    '''
    # Check if we have already completed the signal calculation procedure on this table.
    # There is no easy way to tell from the table itself, so we use an attribute.
    if sig_arr.attrs.complete:
        print("Signal summation array: " + sig_arr.name + " is complete. Continuing...")
        return
    else:
        # Erase the signal table.
        sig_arr[:] = 0

    # Create the array which will store the signal we compute.
    signal = np.zeros(sig_arr.nrows, dtype=np.float32)
    
    # Determine the indices of the rays detected.
    detected_rays = np.where(dt_arr)[0]
    
    # How many indices we need to iterate over.
    max_rays_signal = detected_rays.shape[0]

    # How many batches will we need to run through the entire table
    n_batches = max_rays_signal // max_batch_size
    last_batch_size = max_rays_signal % max_batch_size
    
    # Keep track of the number of rays we have summed.
    rays_summed = 0

    # Perform the source convolution on each batch.
    tqdm_iterator = tqdm.trange(n_batches+1, desc=sig_arr.name + " summing detected rays", leave=True)
    for batch in tqdm_iterator:
        # Check whether we are on the final batch.
        if batch == n_batches:
            # Exit the loop if there is no final batch
            if last_batch_size == 0:
                break
            batch_range = [rays_summed, rays_summed+last_batch_size]
            rays_summed += last_batch_size
        else:
            batch_range = [rays_summed, rays_summed+ max_batch_size]
            rays_summed += max_batch_size
        
        # Get the indices of the detected rays
        detected_ray_coordinate_indices = detected_rays[batch_range[0]:batch_range[1]]

        # Get the coordinate indices of each detected ray.
        coordinate_indices = rt_table.read_coordinates(detected_ray_coordinate_indices, field="coordinate_index")

        # Add the batch signal to the total signal
        np.add.at(signal, coordinate_indices, 1)

    # Write the resulting signal to the database.
    sig_arr[:] = signal

    # The array is complete
    sig_arr.attrs.complete = True
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
    sum_detected_rays(args, settings, rt_table, dt_arr, sig_arr, max_batch_size_sig)
    
    return

def run_simulation_default(args):
    '''
    Run the default simulation using settings.
    '''
    # Get the settings from the config file
    settings = shem.configuration.get_settings(args)
    
    # Get the default parameters from settings.
    parameters = shem.configuration.get_setting_values(settings)
    
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

