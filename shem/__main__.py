#!/usr/bin/env python3

import argparse, textwrap
import os, sys, importlib
import numpy as np
np.random.seed(0)
import cupy as cp
cp.random.seed(0)

import trimesh
import tqdm

import shem.configuration
from shem.definitions import *
import shem.display
import shem.geometry
import shem.mesh
import shem.ray
import shem.scattering
import shem.source
import shem.simulation
import shem.surface
import shem.optimisation

def mesh(args):
    
    if args.verbose:
        print("Creating " + args.type + " mesh...")

    # Create a flat mesh.
    if args.type == 'flat':
        triangles = shem.mesh.create_flat_triangles(args.width, args.height)
    
    # Create a mesh with a sphere on the surface.
    elif args.type == 'sphere':
        triangles = shem.mesh.create_sphere_triangles(args.width, args.height, args.radius, iterations=args.iterations)
    
    # Create a mesh with a rough, randomly generated surface.
    # TODO: Implement rough surface generation.
    elif args.type == 'rough':
        triangles = shem.mesh.create_flat_triangles(args.width, args.height)
    # Create a mesh with a series of trenches
    elif args.type == 'trenches':
        triangles = shem.mesh.create_trenches_triangles(args.width, args.height, args.trench_width, args.trench_height, args.trench_depths)

    # Alter the convention
    if args.xz_convention:
        triangles = np.roll(triangles, -1, axis=-1)

    # Generate the mesh object from the raw triangles.
    mesh = shem.mesh.convert_triangles_to_mesh(triangles)

    # Inspect the mesh before using it. This lead to a segmentation fault and I have no idea why but the mesh saves regardless. Maybe it's because I am using X tunnelling?
    if not args.quiet:
        print("Displaying mesh...")
        scene = trimesh.scene.scene.Scene(mesh)
        scene.show()
        print("Mesh inspection complete.")

    # Create any directories required for the mesh file.
    mesh_file = os.path.join(args.work_dir, args.mesh)
    mesh_directory = os.path.split(mesh_file)[0]
    if not os.path.isdir(mesh_directory):
        if not args.quiet:
            print("Creating directory: " + mesh_directory + "...")
        os.makedirs(mesh_directory)

    # Check if the mesh directory is either empty or has a lock file.
    lock_file = os.path.join(mesh_directory, "shem.lock")
    if len(os.listdir(mesh_directory)) != 0 and not os.path.isfile(lock_file):
        raise OSError("The working directory is not empty and does not have a lock file.")
    # Create a lock file if one does not already exist.
    elif not os.path.isfile(lock_file):
        open(lock_file,'a').close()

    # Save the mesh file, overwriting previous versions
    if not args.quiet and os.path.isfile(mesh_file):
        print("Overwriting " + mesh_file + "...")
    elif args.verbose:
        print("Saving to " + mesh_file + "...")

    mesh.export(mesh_file)

    return

def check_configuration_and_run(args, func):
    # Create the work directory if it doesn't exist already.
    if not os.path.isdir(args.work_dir):
        print("Creating " + args.work_dir)
        os.makedirs(args.work_dir)

    # Check if the working directory is either empty or has a lock file.
    lock_file = os.path.join(args.work_dir, "shem.lock")
    if len(os.listdir(args.work_dir)) != 0 and not os.path.isfile(lock_file):
        raise OSError("The working directory is not empty and does not have a lock file.")
    # Create a lock file if one does not already exist.
    elif not os.path.isfile(lock_file):
        open(lock_file,'a').close()

    if not args.mesh:
        print("Either specify a mesh in .glb format or run the mesh sub-command provided by this tool to generate a mesh to image e.g. python -m shem mesh --help")
        sys.exit()

    # Create the default_config.py file if no configuration file supplied
    config_file = os.path.join(args.work_dir, args.config)
    if not os.path.isfile(config_file):
        print("The configuration file " + config_file +  " does not exist.")
        print("Writing the default configuration to" + config_file + " and exitting. You should check the contents of the config file for more information...")
        shem.configuration.create_default(config_file)
        sys.exit()
    else:
        # Setup the database.
        shem.database.db_setup(args)
        # Run the function.
        func(args)
    return

def run_simulation_default_wrapper(args):
    '''
    Run the simulation using the default settings in the config file.
    '''
    return check_configuration_and_run(args, shem.simulation.run_simulation_default)

def run_analysis_wrapper(args):
    '''
    Run the image analysis program.
    '''
    if not args.image:
        raise ValueError("You have not provided an image to analyse.")
    # We need a png image file
    assert args.image.split('.')[-1] == "png"

    return check_configuration_and_run(args, shem.optimisation.run_analysis)

def main():
    # Main Parser
    parser = argparse.ArgumentParser(prog="SHeM", description=textwrap.dedent('''
    A GPU-accelerated SHeM simulations library.'''), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s 2.1')
    parser.add_argument("-l", "--list-devices", help="list available devices", action="store_true")
    parser.add_argument("-g", "--use-gpu",      help="use the gpu",            action="store_true")
    parser.add_argument("-w", "--work-dir",     help="working directory",      default="./work")
    # We will generate the database from the hash of the config file.
    #parser.add_argument("-d", "--database",     help="h5 database file",       default="simulation.db")
    parser.add_argument("-m", "--mesh",         help="mesh file as stl",       default="mesh.stl")
    # Swap y and z - this software uses the convention that the sample is aligned along the xy plane rather than the xz plane
    parser.add_argument("-L", "--xz-convention", help="scans and meshes aligned along xz direction", action="store_true")
    # Use the database
    parser.add_argument("-D", "--enable-database", help="enable the ability to write to the database AND save outputs", action="store_true")
    
    # Either verbose or quiet, not both
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", help="decrease output verbosity", action="store_true")
    verbosity.add_argument("-q", "--quiet",   help="increase output verbosity", action="store_true")
    
    subparsers = parser.add_subparsers(help='subcommands')

    # Simulation subparser
    parser_gen = subparsers.add_parser('generate', aliases=['gen'], help='create an image using the settings in the config file')
    # Specify a config file to run the simulation and help with analysis
    parser_gen.add_argument("config", help="config file in Python", default="config.py")
    parser_gen.add_argument("-n", "--n-rays", help="number of rays per batch", type=int, default=-1)
    parser_gen.add_argument("-b", "--batches", help="number of batches", type=int, default=-1)
    parser_gen.add_argument("-d", "--disable-write", help="disable the ability to write to the database", action="store_true")
    parser_gen.set_defaults(func=run_simulation_default_wrapper)
    
    parser_ana = subparsers.add_parser('analyse', aliases=['ana'], help='analyse an image based on the settings and paramters template in the config file ')
    # Specify a config file to run the simulation and help with analysis
    parser_ana.add_argument("config", help="config file in Python", default="config.py")
    parser_ana.add_argument("image", help="config file in Python")
    parser_ana.set_defaults(func=run_analysis_wrapper)
    
    # Mesh subparser
    parser_mesh = subparsers.add_parser('mesh', aliases=[], help='create an STL mesh using the command line')
    # Mesh parameters
    parser_mesh.add_argument("-w", "--width",  help="width parameter for meshes", type=float, default=1.0)
    parser_mesh.add_argument("-T", "--height", help="height parameter for meshes", type=float, default=0.2)
    parser_mesh.add_argument("-W", "--trench-width",  help="trench width parameter for meshes", type=float, default=0.8)
    parser_mesh.add_argument("-H", "--trench-height", help="trench height parameter for meshes", type=float, default=0.8)
    parser_mesh.add_argument("-D", "--trench-depths", help="trench depth parameter/s for meshes; can be specified multiple times", type=float, nargs='+', default=[i/5 for i in range(1,5)])
    parser_mesh.add_argument("-r", "--radius", help="radius parameter for spherical meshes", type=float, default=0.1)
    parser_mesh.add_argument("-I", "--iterations", help="iterations parameter for smoothing meshes", type=int, default=4)
    # Mesh type
    parser_mesh.add_argument("-t", "--type", help="mesh type", default="flat")
    parser_mesh.set_defaults(func=mesh)
    
    parser_char = subparsers.add_parser('characterise', aliases=['char'], help='characterise a mesh by calculating the average solid angle visible from each point on its surface')
    parser_char.add_argument("-b", "--batches", help="number of batches", type=int, default=8)
    parser_char.add_argument("-n", "--n-rays", help="number of rays per batch", type=int, default=10e4)
    parser_char.add_argument("-c", "--cull", help="cull these faces from either the start (+) or end (-)", type=int, default=0)
    # Specify a config file to run the simulation and help with analysis
    parser_char.set_defaults(func=shem.simulation.characterise)

    args = parser.parse_args()
    args.func(args)
    
    return

if __name__ == "__main__":
    main()
