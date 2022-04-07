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


# Define the table of rays we are tracking.
class TracedRay(tb.IsDescription):
    coordinate_index = tb.UInt32Col(pos=0) # Index of coordinate tuple defined in coordinates.

    # Ray properties
    a_polar_i  = tb.Float16Col(shape=(2,), pos=1) # Initial ray direction
    a_polar_f  = tb.Float16Col(shape=(2,), pos=2) # Final ray direction
    b_f        = tb.Float16Col(shape=(3,), pos=3) # Final ray origin
    n_scat     = tb.UInt8Col(pos=4)               # Number of scattering events undergone by this ray. This shouldn't be more than 256 in a reasonable world.

# Define a table of file name and file contents hashes.
class FileHash(tb.IsDescription):
    # ID column
    name     = tb.StringCol(itemsize=32,pos=0)
    contents = tb.StringCol(itemsize=32,pos=1)

# Define a table of the hashes of serialised scatteing functions (corresponding to table names) and the serialised functions themselves.
class ScatteringFunction(tb.IsDescription):
    # ID column
    serial_hash = tb.StringCol(itemsize=32,pos=0)
    serial_func = tb.StringCol(itemsize=32,pos=1)

# Define a table of the hashes of serialised source functions (corresponding to table names) and the serialised functions themselves.
class SourceFunction(tb.IsDescription):
    # ID column
    serial_hash = tb.StringCol(itemsize=32,pos=0)
    serial_func = tb.StringCol(itemsize=32,pos=1)

# Define a table of the hashes of serialised detectors and the serialised detectors themselves.
class Detector(tb.IsDescription):
    # ID column
    serial_hash = tb.StringCol(itemsize=32,pos=0)
    shape       = tb.StringCol(itemsize=16,pos=1)
    size        = tb.Float16Col(pos=2)
    position    = tb.Float16Col(shape=(3,),pos=3)
    normal      = tb.Float16Col(shape=(3,),pos=4)

# Define a view of the simulation based on the detector and source function used.
class SimulationView(tb.IsDescription):
    # ID column
    coordinate_index = tb.UInt32Col(pos=0)  # Index of coordinate tuple defined in coordinates.
    detector         = tb.StringCol(itemsize=32,pos=2) # Hash of the detector used.
    source_function  = tb.StringCol(itemsize=32,pos=1) # Hash of the source function used.
    signal           = tb.Float32Col(pos=3) # Strength of the signal.

# https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
# md5 checksum of the contents of fname
def md5_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# md5 hash of a serialised dictionary
def md5_dict(dictname):
    return hashlib.md5(pickle.dumps(dictname)).hexdigest()

def create_default(path):
    """
    Creates the default configuration file at the path supplied.
    """
    return

# Returns the config file loaded as a module.
def load_config(config_file):
    conf = SourceFileLoader("conf", config_file).load_module()
    return conf

def create_new_table(db, group, description, SourceFunction=None):
    # Filter to enable compression on the larger tables.
    io_filter = tb.Filters(complevel=1, complib='lzo', shuffle=True, fletcher32=True)
    
    if SourceFunction is None:
        name = 'default'
    else:
        name = md5_dict(SourceFunction)

    return db.create_table(group, name, description, "Scattering Function Hash: " + name, expectedrows=10**8, filters=io_filter)

# Setup the database we will use to store the results of our computation.
def db_setup(args):
    """
    Setup the database file for running the simulation.
    """

    mesh_file   = os.path.join(args.work_dir, args.mesh)
    config_file = os.path.join(args.work_dir, args.config)
    h5_file     = os.path.join(args.work_dir, args.database)
    
    # Load the config file
    conf = load_config(config_file)

    # Open the database file
    db = tb.open_file(h5_file, mode="a", title="Ray Tracing Database")

    # Create a group to store the details of the simulation.
    sim_group  = db.create_group("/", 'simulation', 'Simulation Results')
    ana_group  = db.create_group("/",   'analysis', 'Simulation Analysis')
    meta_group = db.create_group("/",   'metadata', 'File and Function Metadata')

    # Raw data derived from the simulation using the scattering function provided in the config.
    table_raw_sim_default = create_new_table(db, sim_group,      TracedRay, SourceFunction=None)
    # Analysis of the raw data described above.
    table_raw_ana_default = create_new_table(db, ana_group, SimulationView, SourceFunction=None)
    
    # File hashes to ensure simulation integrity
    table_file_hashes   = db.create_table(meta_group,                'files',           FileHash, "File Hashes")

    # Serialised scattering functions and hashes to track their usage.
    table_scat_hashes   = db.create_table(meta_group, 'scattering_functions', ScatteringFunction, "Serialised Scattering Functions")
    table_source_hashes = db.create_table(meta_group,     'source_functions',     SourceFunction, "Serialised Source Functions")
    table_det_hashes    = db.create_table(meta_group,            'detectors',           Detector, "Detector Properties")

    # Save the file hashes for the mesh and the config file.
    files = [config_file, mesh_file]
    file_hash = table_file_hashes.row
    for f in files:
        file_hash['name'] = hashlib.md5(f.encode()).hexdigest()
        file_hash['contents'] = md5_file(f)
        file_hash.append()
    table_file_hashes.flush()

    db.close()

    return

# Loop until interrupted with Ctrl-C
def trace_rays(
    args,                # Command line arguments.
    db,                  # Database table to save the scan results.
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
    old_rays_traced = len(db)

    # Maximum number of rays we are allowed to trace.
    if save_results and max_scans > 0:
        max_rays_db = coordinates.shape[1] * max_scans
        if old_rays_traced >= max_rays_db:
            print("At least " + str(max_scans) + " scans for scattering function hash " + str(md5_dict(scattering_function)) + " have already been run. Continuing...")
            return
    else:
        max_rays_db = None

    # Number of rays we will have traced when the simulation has been run once.
    rays_traced = old_rays_traced

    # Track which iteration of the loop we are on.
    # We need to use c rather than i because we use i in a for loop later on.
    c = 0
    while (iters < 0 or c < iters) and (max_scans < 0 or rays_traced < max_rays_db) :
        c += 1
        rays_traced += rays_scan

        # We may need to lower the number of rays we trace in a batch so that we have a complete scan without overrunning.
        if max_rays_db is not None and save_results and rays_traced >= max_rays_db:
            # Subtract the number of rays over the maximum specified.
            rays_scan   = max_rays_db - (rays_traced - rays_scan)
            rays_traced = max_rays_db
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
            a = shem.source.direction(a, source_location, source_angle, coordinates, coordinate_indices)
            # Determine the point source origin vector.
            b = shem.source.origin(   b, source_location,               coordinates, coordinate_indices)

            # Scatter the rays off the surface
            rays_f, n_scat = shem.scattering.scatter(rays_i, surface, scattering_function, max_scatters)

            # The data we wish to upload to the database.
            a_polar_f = shem.geometry.cart2polar(                            rays_f[0]                    )[..., 1:]
            b_f       =                                                      rays_f[1]
            # Rotate the direction vectors back to being relative to [0,0,1]
            z = xp.array([0,0,1])
            a_polar_i = shem.geometry.cart2polar(shem.geometry.rotate_frame(-source_location, z, rays_i[0]))[..., 1:]
            # Add back the coordinates we subtracted before.
            a_polar_i[..., 0] += coordinates[2][coordinate_indices]
            a_polar_i[..., 1] += coordinates[3][coordinate_indices]

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
                    ray = db.row
                    ray['coordinate_index'] = coordinate_indices[i]
                    ray['a_polar_i'] =   a_polar_i[i]
                    ray['a_polar_f'] =   a_polar_f[i]
                    ray['b_f']       =         b_f[i]
                    ray['n_scat']    =      n_scat[i]
                
                    ray.append()
                db.flush()

                # Keep track of the number of rays traced this session.
                if args.verbose:
                    print("{:,} complete simulations run, {:,} total rays traced this session, {:} total seconds elapsed...".format(rays_traced // coordinates.shape[1], rays_traced-old_rays_traced, time.time()-start_time), end='\r')

            # If we have already traced the required number of rays, break the loop.
            if max_rays_db is not None and save_results and rays_traced == max_rays_db:
                print("Successfully completed " + str(max_scans) + " scans for scattering function hash " + str(md5_dict(scattering_function)) + ". Continuing...")

        # Exit the loop.
        except KeyboardInterrupt:
            break
    return

# Determine the maximum number of rays which can be traced at once.
def calc_max_rays_scan(
    args,                # Command line arguments.
    db,                  # Database table to save the scan results.
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
                db = db,
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

    # Compression
    # We could fine tune this like we do with the number of rays but I don't see the point as of now...
    io_filter = tb.Filters(complevel=1, complib='lzo', shuffle=True, fletcher32=True)

    # Predetermined data tables
    db_default_sim = db.root.simulation.default
    db_meta_files  = db.root.metadata.files
    db_meta_func   = db.root.metadata.scattering_functions

    # TODO: Verify file hashes here...
    # This really shouldn't be that hard.
    if args.verbose:
        print("Verifying file hashes match those in the database...")
        print(db_meta_files[0])

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
    max_rays_scan = calc_max_rays_scan(
        args = args,
        db = db_default_sim,
        source_location = source_location,
        source_angle = source_angle,
        coordinates = coordinates,
        surface = surface,
        scattering_function = scattering_function,
        max_scatters = max_scatters,
        max_rays_scan_init = max_rays_scan_init,
        iters = 32,
        save_results = False
    )

    # Run the simulation with the greatest possible number of rays
    trace_rays(
        args = args,
        db = db_default_sim,
        source_location = source_location,
        source_angle = source_angle,
        coordinates = coordinates,
        surface = surface,
        scattering_function = scattering_function,
        max_scatters = max_scatters,
        rays_scan = max_rays_scan,
        max_scans = max_scans,
        iters = -1,
        save_results = True
    )

    db.close()

    return
