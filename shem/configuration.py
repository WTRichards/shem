import numpy as np
import cupy as cp
import os,sys

import toml
import trimesh
import tqdm
import math

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

def rays_traced(conf):
    return conf.nparallel*conf.passes*conf.npixels**2

def run_config(args, surface, config_file):
    xp = cp.get_array_module(surface.faces, surface.vertices)
    try:
        conf_dict = toml.load(config_file)
        conf = DictObj(conf_dict)
    except:
        ValueError(config_file + " is not a valid config file")

    # Load in detector details.
    detector_names = []
    detector_locations = []
    detector_radii = []
    for detector, parameters in conf.detector.items():
        detector_names.append(detector)
        detector_locations.append(parameters.location)
        detector_radii.append(parameters.radius)

    # Single source direction
    source        = xp.array(conf.source.location)
    source_radius = conf.source.radius

    # Multiple detector directions
    detectors        = xp.array(detector_locations)
    detectors_radius = xp.array(detector_radii, dtype=xp.float32)
    detectors_dim    = detectors_radius.size

    if args.index_method:
        detection_method = 'index'
    else:
        detection_method = 'matrix'

    # Define the parameters of the scan.
    # TODO: Enable scans over arbitrary combinations of variables.
    x_dim, y_dim, theta_dim, phi_dim = 1, 1, 1, 1
    axis_vals = ['x','y','theta','phi', 'None']

    if conf.scan.x_axis not in axis_vals or conf.scan.y_axis not in axis_vals:
        raise ValueError("Axis values must be one of:", axis_vals)
    if conf.scan.x_axis == 'x'     or conf.scan.y_axis == 'x':
        x_dim = conf.npixels
    if conf.scan.x_axis == 'y'     or conf.scan.y_axis == 'y':
        y_dim = conf.npixels
    if conf.scan.x_axis == 'theta' or conf.scan.y_axis == 'theta':
        theta_dim = conf.npixels
    if conf.scan.x_axis == 'phi'   or conf.scan.y_axis == 'phi':
        phi_dim = conf.npixels

    x_     = xp.linspace(-conf.scan.x.range/2, +conf.scan.x.range/2, x_dim)
    y_     = xp.linspace(-conf.scan.y.range/2, +conf.scan.y.range/2, y_dim)
    theta_ = xp.linspace(-conf.scan.theta.range/2, +conf.scan.theta.range/2, theta_dim)
    phi_   = xp.linspace(-conf.scan.phi.range/2, +conf.scan.phi.range/2, phi_dim)
    
    # Just x and y for now.
    # Create the displacement vectors corresponding to the scan
    # This is a grid in the xy plane centered at the origin.
    displacement_full = xp.empty((x_dim, y_dim, 3), dtype=xp.float32)
    yv, xv = xp.meshgrid(x_, y_, sparse=True)
    if x_dim != 1:
        displacement_full[:, :, 0] = xv
    if y_dim != 1:
        displacement_full[:, :, 1] = yv
    displacement_full[:, :, 2] = 0

    displacement_full += xp.array([conf.scan.x_shift,conf.scan.y_shift,conf.scan.z_shift])

    # Create the array holding the detector information.
    scan_data = xp.zeros((detectors_dim, x_dim, y_dim), dtype=xp.float32)

    # Split into pieces
    displacements = xp.split(displacement_full, 2**(math.floor(math.log2(conf.blocks))))
    block_size = conf.npixels // 2**(math.floor(math.log2(conf.blocks)))

    # Allocate memory for rays.
    # We decompose the system into strips along y
    rays = xp.empty((2, block_size, y_dim, conf.nparallel, 3), dtype=xp.float32)

    # Repeat the simulation a number of times.
    scan_loop = tqdm.trange(conf.passes*len(displacements), leave=False)
    for d, displacement in enumerate(displacements):
        for p in range(conf.passes):
            # Seed random data repeatedly to prevent weirdness and improve consistency.
            xp.random.seed(d*p + p)
            # Disable NumPy warnings - we avoid performing certain checks on the GPU for speed.
            np.seterr(all="ignore")
            # Since the system is linear, we can calculate the effect of each scattering distribution separately and sum, weighted by their respective strengths.
            rays = shem.source.superposition(rays, displacement, source, source_radius, conf.source.function)
            scan_data[:, d*block_size:(d+1)*block_size, :] += conf.scattering.diffuse.strength  * shem.scattering.diffuse( rays, surface, conf.scattering.n, detectors, detectors_radius, displacement, method=detection_method)
            scan_data[:, d*block_size:(d+1)*block_size, :] += conf.scattering.specular.strength * shem.scattering.specular(rays, surface, conf.scattering.n, detectors, detectors_radius, displacement, method=detection_method)
            scan_loop.update()
    scan_loop.close()

    # Normalise the scan data based on the relative strengths of the scattering functions - the numbers are affected by the weights.
    
    # Retrieve data from the GPU if necessary
    if args.use_gpu:
        scan_data_ = scan_data.get()
    else:
        scan_data_ = scan_data

    # Save the image
    for n, name in enumerate(detector_names):
        shem.display.save_image(scan_data_[n], config_file.split('.')[0] + '-detector-' + name + '.png')

    return
