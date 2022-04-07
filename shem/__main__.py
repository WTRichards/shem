#!/usr/bin/env python3

import argparse, textwrap
import os, sys, importlib
import numpy as np
import cupy as cp

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
import shem.surface

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

    # Generate the mesh object from the raw triangles.
    mesh = shem.mesh.convert_triangles_to_mesh(triangles)

    # Inspect the mesh before using it.
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

def generate(args):
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
    if not args.config:
        print("No configuration file supplied.")
        default_config_file = os.path.join(args.work_dir, "default_config.py")
        if os.path.isfile(default_config_file):
            print("Check for the default configuration in your work directory.")
        else:
            print("Generating default_config.py and exiting.")
            shem.configuration.create_default(default_config_file)
        sys.exit()
    else:
        config_file = os.path.join(args.work_dir, args.config)
        db_file     = os.path.join(args.work_dir, args.database)
        # Check that the config file supplied on the command line exists.
        if not os.path.isfile(config_file):
            raise ValueError("Config file: " + config_file + " does not exist.")
        if not os.path.isfile(db_file):
            # Setup the database
            shem.configuration.db_setup(args)
        # Run the simulation in the default mode.
        shem.configuration.generate(args)
    return

def analyse(args):
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
    if not args.config:
        print("No configuration file supplied.")
        default_config_file = os.path.join(args.work_dir, "default_config.py")
        if os.path.isfile(default_config_file):
            print("Check for the default configuration in your work directory.")
        else:
            print("Generating default_config.py and exiting.")
            shem.configuration.create_default(default_config_file)
        sys.exit()
    else:
        config_file = os.path.join(args.work_dir, args.config)
        db_file     = os.path.join(args.work_dir, args.database)
        # Check that the config file supplied on the command line exists.
        if not os.path.isfile(config_file):
            raise ValueError("Config file: " + config_file + " does not exist.")
        if not os.path.isfile(db_file):
            # Setup the database
            shem.configuration.db_setup(args)
        # Begin the analysis
        shem.configuration.run_analysis(args)
    return

def main():
    # Main Parser
    parser = argparse.ArgumentParser(prog="SHeM", description=textwrap.dedent('''
    A GPU-accelerated SHeM simulations library.'''), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s 2.1')
    parser.add_argument("-l", "--list-devices", help="list available devices", action="store_true")
    parser.add_argument("-g", "--use-gpu",      help="use the gpu",            action="store_true")
    parser.add_argument("-w", "--work-dir",     help="working directory",      default="./work")
    parser.add_argument("-d", "--database",     help="h5 database file",       default="simulation.db")
    parser.add_argument("-m", "--mesh",         help="mesh file as glb",       default="mesh.glb")
    
    # Either verbose or quiet, not both
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", help="decrease output verbosity", action="store_true")
    verbosity.add_argument("-q", "--quiet",   help="increase output verbosity", action="store_true")
    
    subparsers = parser.add_subparsers(help='subcommands')

    # Simulation subparser
    parser_gen = subparsers.add_parser('generate', aliases=['gen'], help='image generation subcommand')
    # Specify a config file to run the simulation and help with analysis
    parser_gen.add_argument("config", help="config file in Python", default="default_config.py")
    parser_gen.set_defaults(func=generate)
    
    parser_ana = subparsers.add_parser('analyse', aliases=['ana'], help='image analysis subcommand')
    # Specify a config file to run the simulation and help with analysis
    parser_ana.add_argument("config", help="config file in Python", default="default_config.py")
    parser_ana.set_defaults(func=analyse)
    
    # Mesh subparser
    parser_mesh = subparsers.add_parser('mesh', aliases=[], help='mesh creation subcommand')
    # Mesh parameters
    parser_mesh.add_argument("-W", "--width",  help="width parameter for meshes", type=float, default=1.0)
    parser_mesh.add_argument("-H", "--height", help="height parameter for meshes", type=float, default=0.2)
    parser_mesh.add_argument("-R", "--radius", help="radius parameter for spherical meshes", type=float, default=0.1)
    parser_mesh.add_argument("-I", "--iterations", help="iterations parameter for smoothing meshes", type=int, default=4)
    # Mesh type
    parser_mesh.add_argument("-t", "--type", help="mesh type", default="flat")
    parser_mesh.set_defaults(func=mesh)

    args = parser.parse_args()
    args.func(args)
    
    return

if __name__ == "__main__":
    main()
