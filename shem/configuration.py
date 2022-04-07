import numpy as np
# Disable NumPy warnings - we avoid performing certain checks on the GPU for speed.
np.seterr(all="ignore")
import cupy as cp
import tables as tb
import os,sys

import trimesh
import tqdm
import math
import json
import time

from importlib.machinery import SourceFileLoader
import hashlib
import pickle

from   shem.definitions import *
import shem.display
import shem.geometry
import shem.mesh
import shem.ray
import shem.scattering
import shem.source

# TODO: Implement the ability to create a default configuration file.
# TODO: Setup Cerberus input validator.

def create_default(path):
    """
    Creates the default configuration file at the path supplied.
    """
    return

# Returns the config file loaded as a module.
def load_config(config_file):
    conf = SourceFileLoader("conf", config_file).load_module()
    return conf


# Loop until interrupted with Ctrl-C
def trace_rays(
    args,                # Command line arguments.
    rt_table,                  # Database table to save the scan results.
    source_location,     # Location of point ray source.
    source_angle,        # Angular diameter of the point source. Convolve with source function later.
    coordinates,         # 4 x n array of coordinates in (x, y, theta, phi) to scan.
    surface,             # The surface object.
    scattering_function, # Scattering function defined in a dictionary.
    max_scatters,        # Maximum number of times a ray can be scattered.
    rays_scan,           # number of rays to scan per batch.
    max_scans=-1,        # maximum number of scans to run through for a particular configuration. Set to -1 to run forever
    iters=-1,            # number of times to run through the main loop. Useful for calibration. Set to -1 to run forever.
    save_results=True    # save the results of the calculations performed to the database. Useful for calibration.
    ):
    
    if args.use_gpu:
        xp = cp
    else:
        xp = np
    
    # Allocate memory for the rays.
    # We do this once at the start for performance.
    # If this fails, just return so that we can figure out an appropriate value for max_rays without crashing the program.

    rays_i = xp.empty((2, rays_scan, 3), dtype=xp.float32)
    a = rays_i[0]
    b = rays_i[1]
   
    if args.verbose:
        print("Running until interrupted with Ctrl-C...")
    
    # Track when we start.
    start_time = time.time()
    
    # Number of rays in the database.
    old_rays_traced = len(rt_table)

    # Maximum number of rays we are allowed to trace.
    if save_results and max_scans > 0:
        max_rays_rt_table = coordinates.shape[1] * max_scans
        if old_rays_traced >= max_rays_rt_table:
            print("At least " + str(max_scans) + " scans for scattering function hash " + str(hash_obj(scattering_function)) + " have already been run. Continuing...")
            return
    else:
        max_rays_rt_table = None

    # Number of rays we will have traced when the simulation has been run once.
    rays_traced = old_rays_traced

    # Track which iteration of the loop we are on.
    # We need to use c rather than i because we use i in a for loop later on.
    c = 0
    while (iters < 0 or c < iters) and (max_scans < 0 or rays_traced < max_rays_rt_table) :
        c += 1
        rays_traced += rays_scan

        # We may need to lower the number of rays we trace in a batch so that we have a complete scan without overrunning.
        if max_rays_rt_table is not None and save_results and rays_traced >= max_rays_rt_table:
            # Subtract the number of rays over the maximum specified.
            rays_scan   = max_rays_rt_table - (rays_traced - rays_scan)
            rays_traced = max_rays_rt_table
            # Reallocate the memory required to run the final batch.
            del rays_i
            rays_i = xp.empty((2, rays_scan, 3), dtype=xp.float32)
            a = rays_i[0]
            b = rays_i[1]
        
        # Determine the coordinate indices of the next batch of rays to trace.
        # We take this modulo the number of coordinates so we can loop back round to the beginning without stopping.
        coordinate_indices = xp.arange(rays_traced-rays_scan, rays_traced) % coordinates.shape[1]
       
        # Break out of the loop if we hit Ctrl-C.
        try:
            # Determine the point source direction vector.
            # We will trace rays in a cone defined by source_angle and convolve with various source distributions later.
            # We also want the angular distribution relative to the source vector for analysis later, so return that too. 
            a, a_polar_i = shem.source.direction(a, source_location, source_angle, coordinates, coordinate_indices)
            # Determine the point source origin vector.
            b = shem.source.origin(   b, source_location,               coordinates, coordinate_indices)

            # Scatter the rays off the surface
            rays_f, n_scat = shem.scattering.scatter(rays_i, surface, scattering_function, max_scatters)

            # The data we wish to upload to the database.
            a_polar_f = shem.geometry.cart2polar(                            rays_f[0]                    )[..., 1:]
            b_f       =                                                      rays_f[1]

            # Move the data off the GPU
            if args.use_gpu:
                a_polar_i = a_polar_i.get()
                a_polar_f = a_polar_f.get()
                b_f       =       b_f.get()
                n_scat    =    n_scat.get()
                coordinate_indices = coordinate_indices.get()

            # Save results to the database.
            if save_results:
                # Try and look for a better way to do this and parallelise.
                for i in range(rays_scan):
                    ray = rt_table.row
                    ray['coordinate_index'] = coordinate_indices[i]
                    ray['a_polar_i'] =   a_polar_i[i]
                    ray['a_polar_f'] =   a_polar_f[i]
                    ray['b_f']       =         b_f[i]
                    ray['n_scat']    =      n_scat[i]
                
                    ray.append()
                rt_table.flush()

                # Keep track of the number of rays traced this session.
                if args.verbose:
                    print("{:,} complete simulations run, {:,} total rays traced this session, {:} total seconds elapsed...".format(rays_traced // coordinates.shape[1], rays_traced-old_rays_traced, time.time()-start_time), end='\r')

            # If we have already traced the required number of rays, break the loop.
            if max_rays_rt_table is not None and save_results and rays_traced == max_rays_rt_table:
                print("Successfully completed " + str(max_scans) + " scans for scattering function hash " + str(hash_obj(scattering_function)) + ". Continuing...")

        # Exit the loop.
        except KeyboardInterrupt:
            break
    return

# Determine the maximum number of rays which can be traced at once.
def calc_max_rays_scan(
    args,                # Command line arguments.
    rt_table,                  # Database table to save the scan results.
    source_location,     # Location of point ray source.
    source_angle,        # Angular diameter of the point source. Convolve with source function later.
    coordinates,         # 4 x n array of coordinates in (x, y, theta, phi) to scan.
    surface,             # The surface object.
    scattering_function, # Scattering function defined in a dictionary.
    max_scatters,        # Maximum number of times a ray can be scattered.
    max_rays_scan_init=1024, # number of rays to start the calibration with.
    max_scans=-1,        # maximum number of scans to run through for a particular configuration. Set to -1 to run forever
    iters=256,           # number of times to run through the main loop. Useful for calibration. Set to -1 to run forever
    save_results=False,  # save the results of the calculations performed to the database. Useful for calibration.
    r = 1.01,            # multiplicative increase to be applied to rays_scan at each iteration.
    final_reduction=1/50 # the loop stops when it can't allocate the necessary memory. Reduce this final value by this fraction.
    ):

    max_rays_scan = max_rays_scan_init

    try:
        while True:
            start_time = time.time()
            trace_rays(
                rays_scan = max_rays_scan,
                iters = iters,
                save_results = save_results,
                args = args,
                rt_table = rt_table,
                source_location = source_location,
                source_angle = source_angle,
                coordinates = coordinates,
                surface = surface,
                scattering_function = scattering_function,
                max_scatters = max_scatters,
                max_scans = max_scans
            )
            end_time = time.time()
            time_elapsed = end_time - start_time
            

            if not args.quiet:
                print("max_rays_scan = {:,}, time elapsed per ray = {:} ns".format(max_rays_scan, 10**9 * time_elapsed/(iters*max_rays_scan)))

            # Increase the maximum number of rays
            max_rays_scan *= r
            max_rays_scan = math.floor(max_rays_scan)
    except:
        # Reduce the value of max_rays_scan which crashes by a the final_reduction provided.
        max_rays_scan *= (1.0 - final_reduction)
        max_rays_scan = math.floor(max_rays_scan)
    
    if not args.quiet:
        print("Optimum max_rays_scan = {:,}".format(max_rays_scan))

    return max_rays_scan

def generate(args):
    # Get absolute file paths
    mesh_file   = os.path.join(args.work_dir, args.mesh)
    config_file = os.path.join(args.work_dir, args.config)
    h5_file     = os.path.join(args.work_dir, args.database)
    
    # Load the config file
    conf = load_config(config_file)

    # Open the database file.
    db = tb.open_file(h5_file, mode="r+")

    # Predetermined data structures
    rt_default_table = db.root.rt.default  # The default ray tracing database
    dt_default_arr   = db.root.dt.default  # The default ray detection array
    sig_default_arr  = db.root.sig.default # The default detected signal array
    meta_files_table = db.root.metadata.files # File metadata (hashes)

    # TODO: Verify file hashes here...
    # This really shouldn't be that hard.
    if args.verbose:
        print("Verifying file hashes match those in the database...")
        print(meta_files_table[0])

    # Extract the combined meshes from the file
    mesh = trimesh.load_mesh(mesh_file).dump(concatenate=True)

    # Load the meshes into a surface object which stores its data either on the CPU or GPU.
    # Define xp as either NumPy or CuPy, depending on the same.
    if args.use_gpu:
        xp = cp
    else:
        xp = np

    try:
        properties, per_face_props = conf.surface_properties(mesh.vertices, mesh.faces)
    except:
        properties, per_face_props = None, None
        print("surface_properties function not defined. Continuing without any known surface properties")

    # Create the surface object
    surface = shem.surface.Surface(vertices = mesh.vertices,
                                         faces = mesh.faces,
                                    properties = properties,
                                per_face_props = per_face_props,
                                            xp = xp)

    # Apply coordinate shifts to the surface.
    try:
        surface.shift_x(conf.x_shift)
    except:
        pass
    try:
        surface.shift_y(conf.y_shift)
    except:
        pass
    try:
        surface.shift_z(conf.z_shift)
    except:
        pass

    # Load in the variable definitions from the config file.
    # There is probably a more elegant way to do this but this is pretty general and does the job.

    try:
        scattering_function = conf.scattering_function
    except:
        raise ValueError("scattering_function not specified. For futher information on how you should specify the scattering function, see the example configuration file.")

    try:
        source_location     = xp.array(conf.source_location)
    except:
        raise ValueError("source_location not specified. source_location is a 3 vector representing the point ray source relative to the origin.")

    try:
        coordinates = xp.array(conf.coordinates)
    except:
        raise ValueError("coordinates not specified. The variable coordinates is a 4 x n array containing x, y, theta and phi coordinates to be scanned over.")

    try:
        max_rays_scan_init = conf.max_rays_scan_init
    except:
        max_rays_scan_init = 2**10
        print("max_rays_scan_init not specified. Using an initial guess of " + str(max_rays_scan_init) + " to calibrate the maximum number of rays which this system can trace in one go...")
    
    try:
        max_scans = conf.max_scans
    except:
        max_scans = -1
        print("max_scans not specified. Specify to limit the number of scans over the coordinates. Running indefinitely until stopped...")

    try:
        source_angle = conf.source_angle
    except:
        source_angle = np.radians(1)
        print("source_angle not specified. This variable is used to determine the size of the cone rays are generated in. The source distribution is determined later by giving each ray an appropriate weight based on its angle relative to a 'perfect ray' so this describes a windowing function convolved with whatever source distribution you use later on. Using a sensible default of " + str(np.degrees(source_angle)) + " degrees...")

    try:
        max_scatters = math.floor(conf.max_scatters)
    except:
        max_scatters = 2**3
        print("max_scatters not specified. This variable determines the number of times each ray can be scattered. Using a sensible default of " + str(max_scatters) + "...")
    if max_scatters < 0 or max_scatters > 255:
        max_scatters = 255
        print("max_scatters must be between 0 and 255. Setting to max_scatters = 255...")

    
    # Calculate the maximum number of rays we can trace in one batch for this simulation.
    # The results will not be saved. If this takes too long, consider setting iters to be a smaller value.
    if args.use_gpu:
        max_rays_scan = calc_max_rays_scan(
            args = args,
            rt_table = rt_default_table,
            source_location = source_location,
            source_angle = source_angle,
            coordinates = coordinates,
            surface = surface,
            scattering_function = scattering_function,
            max_scatters = max_scatters,
            max_rays_scan_init = max_rays_scan_init,
            iters = 32,
            save_results = False,
        )
    else:
        max_rays_scan = 2**8

    # Trace and scatter the rays off the surface
    trace_rays(
        args = args,
        db = rt_default_table,
        source_location = source_location,
        source_angle = source_angle,
        coordinates = coordinates,
        surface = surface,
        scattering_function = scattering_function,
        max_scatters = max_scatters,
        rays_scan = max_rays_scan,
        max_scans = max_scans,
        iters = -1,
        save_results = True,
    )

    # Detect the rays which were scattered.
    """
    detect_rays(
        args = args,
        rt_table = rt_default_table,
        dt_arr   = dt_default_arr,
        detector_location = detector_location,
        detector_normals = detector_normal,
        detector_radius = detector_radius,
        coordinates = coordinates,
        save_results = True,
    )

    # Convolve the rays with the source function and sum over each index, saving the result.
    source_convolve_rays(
        args = args,
        rt_table = rt_default_table,
        dt_arr   = dt_default_arr,
        sig_arr  = sig_default_arr,
        arr_sig = db_default_sig,
        source_function = source_function,
        save_results = True,
    )
    """

    db.close()

    return
