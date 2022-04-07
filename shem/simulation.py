import tables

import numpy as np
import cupy  as cp

from shem.definitions import *

# Define the table of rays we are tracking.
class ScatteredRay(tb.IsDescription):
    coordinate_index = tb.UInt32Col(pos=0) # Index of coordinate tuple defined in coordinates.

    # Ray properties
    a_polar_i  = tb.Float16Col(shape=(2,), pos=1) # Initial ray direction
    a_polar_f  = tb.Float16Col(shape=(2,), pos=2) # Final ray direction
    b_f        = tb.Float16Col(shape=(3,), pos=3) # Final ray origin
    n_scat     = tb.UInt8Col(pos=4)               # Number of scattering events undergone by this ray. This shouldn't be more than 256 in a reasonable world.
    #detected   = tb.BoolCol(pos=5,dflt=False)     # Sets whether the ray has been detected. Defaults to False.

# Define a table of file name and file contents hashes.
class FileHash(tb.IsDescription):
    # ID column
    name     = tb.StringCol(itemsize=32,pos=0) # Hash of the file name. This lets us support arbitrarily long file names.
    contents = tb.StringCol(itemsize=32,pos=1) # Hash of the file contents.

def create_new_simulation_db(db, conf, parameters=None):
    # Filter to enable compression
    io_filter = tb.Filters(complevel=1, complib='lzo', shuffle=True, fletcher32=True)

    # Name the arrays and tables after the hash of their parameters
    if parameters is None:
        name = 'default'
    else:
        name = hash_obj(parameters)
    
    # Expected sizes
    n_coords = conf.coordinates.shape[1]
    n_rows   = n_coords * conf.max_scans

    # Create the table which stores the result of the ray tracing.
    rt_table = db.root.rt.create_table(name, ScatteredRay, "Parameters Hash: " + name, expectedrows=n_rows, filters=io_filter)
    rt_table.attrs.parameters = parameters["scattering"]

    # Create the array which stores which rays were detected.
    dt_atom = tb.BoolAtom(dflt=False)
    dt_arr = db.root.dt.create_carray(name, dt_atom, (n_rows,), fiters=io_filter)
    dt_arr.attrs.parameters = parameters["detection"]
    
    # Create the array which stores the per coordinate signal detected.
    sig_atom = tb.Float64Atom()
    sig_arr = db.root.sig.create_carray(name, sig_atom, (n_coords,), filters=io_filter)
    sig_arr.attrs.parameters = parameters["source"]

    return rt_table, dt_arr, sig_arr

# Setup the database we will use to store the results of our computation.
def db_setup(args):
    '''
    Setup the database file for running the simulation.
    '''

    # File locations
    mesh_file   = os.path.join(args.work_dir, args.mesh)
    config_file = os.path.join(args.work_dir, args.config)
    h5_file     = os.path.join(args.work_dir, args.database)

    # Load the config file
    conf = load_config(config_file)

    # Open the database file
    db = tb.open_file(h5_file, mode="a", title="Ray Tracing Database")

    # Create the groups under which the information we record from the simulation will be stored.
    rt_group   = db.create_group("/", 'rt',       'Ray Tracing')
    dt_group   = db.create_group("/", 'dt',       'Rays Detected')
    sig_group  = db.create_group("/", 'signal',   'Signal Detected')

    # Create a group for metadata
    meta_group = db.create_group("/", 'metadata', 'Simulation Metadata')

    # Create the tables and carrays which will store the simulation data.
    create_new_simulation_db(db, conf, parameters=None)

    # File hashes to ensure simulation integrity
    file_hashes_table   = db.root.metadata.create_table('files', FileHash, "File Hashes")

    # Save the file hashes for the mesh and the config file.
    files = [config_file, mesh_file]
    file_hash = file_hashes_table.row
    for f in files:
        file_hash['name'] = hashlib.md5(f.encode()).hexdigest()
        file_hash['contents'] = md5_file(f)
        file_hash.append()
    file_hashes_table.flush()

    # Close the database
    db.close()

    return

