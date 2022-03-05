#!/usr/bin/env python3

import argparse, textwrap
import os, sys, importlib

import tqdm

import shem.configuration
import shem.display
import shem.geometry
import shem.mesh_manipulation
import shem.ray_tracing
import shem.scattering_functions
import shem.source_functions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="SHeM", description=textwrap.dedent('''
    A GPU-accelerated SHeM simulations library based on PyTorch.'''), formatter_class=argparse.RawTextHelpFormatter)

    # Either verbose or quiet, not both
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose",        help="decrease output verbosity", action="store_true")
    verbosity.add_argument("-q", "--quiet",          help="increase output verbosity", action="store_true")
    
    # Run once then exit
    parser.add_argument("-l", "--list-devices",   help="list available devices",    action="store_true")
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    # Specify a working directory
    parser.add_argument("-w", "--work-dir", help="working directory", default="./")
    # Specify a device to use - CPU or GPU,
    parser.add_argument("-d", "--device", help="device target (CPU or GPU)", default="CPU")
    # Can run multiple config files, using the default if none are specified
    parser.add_argument("config", nargs='*', help="config files written in yaml format")

    args = parser.parse_args()

    device = args.device

    # Check if the working directory is either empty or has a lock file.
    lock_file = os.path.join(args.work_dir, "shem.lock")
    if len(os.listdir(args.work_dir)) != 0 and not os.path.isfile(lock_file):
        raise OSError("The working directory is not empty and does not have a lock file.")
    # Create a lock file if one does not already exist.
    elif not os.path.isfile(lock_file):
        open(lock_file,'a').close()

    # Use the default configuration if no command line argument is supplied.
    if not args.config:
        print("No configuration file supplied.")
        default_config_file = os.path.join(args.work_dir, "default_config.yaml")
        if os.path.isfile(default_config_file):
            print("Check for the default configuration in your work directory.")
        else:
            print("Generating default_config.yaml and exiting.")
            shem.configuration.create_default(default_config_file)
    else:
        # Check each config file supplied on the command line exists.
        for config_file in args.config:
            if not os.path.isfile(config_file):
                raise ValueError("Config file: " + config_file + " does not exist.")
        # Import each config file sequentially as a Python module and pass to do_something():
        if not args.quiet:
            config_loop = tqdm.trange(len(args.config))
            for config_file in args.config:
                config_loop.set_description("Parsing " + config_file)
                shem.configuration.run_config(args, config_file, device)
                config_loop.update()
            config_loop.close()
        else:
            for config_file in args.config:
                shem.configuration.run_config(args, config_file, device)

