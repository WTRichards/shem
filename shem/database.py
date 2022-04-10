import os
import time
import tables as tb

import numpy as np
import cupy  as cp

from shem.definitions import *
import shem.configuration

# Filter to enable compression.
# We could do clever things like optimise for the best compression type but this is good enough (which is pretty good!).
io_filter = tb.Filters(complevel=1, complib='lzo', shuffle=True, fletcher32=True)

class TracedRay(tb.IsDescription):
    '''
    The information we record about the rays we trace
    '''
    coordinate_index = tb.UInt32Col(pos=0) # Index of coordinate tuple defined in coordinates.

    # Ray properties
    a_polar_i  = tb.Float32Col(shape=(2,), pos=1) # Initial ray direction
    a_polar_f  = tb.Float32Col(shape=(2,), pos=2) # Final ray direction
    b_f        = tb.Float32Col(shape=(3,), pos=3) # Final ray origin
    n_scat     = tb.UInt8Col(pos=4)               # Number of scattering events undergone by this ray. This shouldn't be more than 256 in a reasonable world.

# Define a table of file name and file contents hashes.
class FileHash(tb.IsDescription):
    # ID column
    name     = tb.StringCol(itemsize=32,pos=0) # Hash of the file name. This lets us support arbitrarily long file names.
    contents = tb.StringCol(itemsize=32,pos=1) # Hash of the file contents.

def get_name(parameters):
    return "hash_" + hash_obj(parameters)

def create_new_simulation_db(db, settings, parameters):

    # Name the arrays and tables after the hash of their parameters.
    name = get_name(parameters)
    
    # Expected sizes
    n_coords =            settings["meta"]["coordinates"].shape[1]
    n_rows   = n_coords * settings["meta"][  "max_scans"]

    # Create the table which stores the result of the ray tracing.
    rt_table = db.create_table(db.root.rt, name, TracedRay, "Parameters Hash: " + name, expectedrows=n_rows, filters=io_filter)

    # Create the array which stores the indices of the rays which were detected.
    dt_atom = tb.BoolAtom(dflt=False)
    dt_arr = db.create_carray(db.root.dt, name, dt_atom, (n_rows,), filters=io_filter)
    dt_arr.attrs.complete = False
    
    # Create the array which stores the per coordinate signal detected.
    sig_atom = tb.Float64Atom()
    sig_arr = db.create_carray(db.root.sig, name, sig_atom, (n_coords,), filters=io_filter)
    sig_arr.attrs.complete = False

    return rt_table, dt_arr, sig_arr

def create_database_file(args, title="Ray Tracing Database"):
    h5_file = os.path.join(args.work_dir, args.database)
    db = tb.open_file(h5_file, mode="a", title=title)
    return db

def open_database_file(args, mode="r"):
    h5_file = os.path.join(args.work_dir, args.database)
    db = tb.open_file(h5_file, mode=mode)
    return db

# Setup the database we will use to store the results of our computation.
def db_setup(args):
    '''
    Setup the database file for running the simulation.
    '''
    
    # Get the settings from the config file.
    settings = shem.configuration.get_settings(args)

    # Create the database file.
    db = create_database_file(args, title="Ray Tracing Database")

    # Create the groups under which the information we record from the simulation will be stored.
    rt_group   = db.create_group("/", 'rt',   'Ray Tracing')
    dt_group   = db.create_group("/", 'dt',  'Rays Detected')
    sig_group  = db.create_group("/", 'sig', 'Signal Detected')

    # Create a group for metadata
    meta_group = db.create_group("/", 'metadata', 'Simulation Metadata')

    # Parameters array to keep track of which simulations we have run.
    #parameters_atom = tb.ObjectAtom()
    # This doesn't need to be a vlarray. We can get the parameters from the config file as a vector (eventually). For now, we'll skip it.
    #parameters_arr = db.create_vlarray(db.root.metadata, 'parameters', parameters_atom, "Simulation Parameters", expectedrows=2**10, filters=io_filter)
    #parameters_arr = db.create_vlarray(db.root.metadata, 'parameters', parameters_atom, "Simulation Parameters", expectedrows=2**10)
    
    # File hashes to ensure simulation integrity
    #file_hashes_table = db.create_table(db.root.metadata, 'files', FileHash, "File Hashes")

    # Close the database
    db.close()

    return
