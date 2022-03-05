import toml
import os

import numpy as np
import trimesh

import shem.geometry
import shem.mesh_manipulation
import shem.source_functions

ScanTypes = {'point', 'line', 'plane'}
MeshTypes  = {'flat', 'sphere', 'rough'}
SourceFunctions = {
        'delta': shem.source_functions.delta,
        'delta_': shem.source_functions.delta_,
        }

# TODO: Implement the ability to create a default configuration file.
# TODO: Setup Cerberus input validator.
def create_default(path):
    '''
    Creates the default configuration file at the path supplied.
    '''
    return

def get_conf_val(conf, *keys):
    '''
    Accepts a (potentially nested) dictionary and a variable number of keys.
    Returns the value corresponding to the first key in the dictionary or None.
    '''
    for key in keys:
        if key in conf.keys():
            return conf[key]
    return None

def reshape_rays(rays, ppixels, npixels, dims=2, flatten=True):
    '''
    Accepts:
        rays - A 2 (x npixel (x npixel)) x ppixel x 3 NumPy array
        ppixels - Integer. Number of rays used per pixel rendered.
        npixels - Integer. Number of points imaged along any particular direction.
        dims - Integer. Dimensionality (point, line or plane).
        flatten - Boolean. Can either flatten or restore the shape of the array.
    Returns:
        An array reshaped to either ? x 3 or restored to its original form.
    '''
    return

def run_config(args, config_file, device):
    '''
    Executes the provided toml config on the provided device.
    '''

    try:
        conf = toml.load(config_file)
    except ValueError:
        print(config_file + " is not a valid config file")

    if not conf is None:
        if args.verbose:
            print("Applying configuration...")
        title       = get_conf_val(conf, 'title')
        description = get_conf_val(conf, 'description')
        sim         = get_conf_val(conf, 'simulation')

    if not sim is None:
        if args.verbose:
            print("  Applying simulation configuration...")
        npixels  = get_conf_val(sim, 'npixels')
        ppixels  = get_conf_val(sim, 'ppixels', 'rays_per_pixel', 'raysperpixel')
        mesh     = get_conf_val(sim, 'mesh')
        scat     = get_conf_val(sim, 'scattering', 'scat')
        scan     = get_conf_val(sim, 'scan', 'scanning')
        source   = get_conf_val(sim, 'source')
        detector = get_conf_val(sim, 'detector')
   
    if not mesh is None:
        if args.verbose:
            print("    Applying mesh configuration...")
        create_mesh  = get_conf_val(mesh, 'create')
        inspect_mesh = get_conf_val(mesh, 'inspect')
        mesh_file    = get_conf_val(mesh, 'file')
        mesh_type    = get_conf_val(mesh, 'type')
        width        = get_conf_val(mesh, 'width')
        height       = get_conf_val(mesh, 'height')
        radius       = get_conf_val(mesh, 'radius')

    if not scat is None:
        if args.verbose:
            print("    Applying scattering distribution configuration...")
        specular = get_conf_val(scat, 'specular', 'spec')
        if not specular is None:
            k_s       = get_conf_val(specular, 'strength', 'k')
            shininess = get_conf_val(specular, 'shininess')
        diffuse = get_conf_val(scat, 'Lambertian', 'Lambert', 'diffuse', 'cosine')
        if not diffuse is None:
            k_d = get_conf_val(diffuse, 'strength', 'k')
        ambient  = get_conf_val(scat, 'ambient')
        if not ambient is None:
            k_a = get_conf_val(ambient, 'strength', 'k')

    if not scan is None:
        if args.verbose:
            print("    Applying scan configuration...")
        scan_type   = get_conf_val(scan, 'scan_type', 'type')
        scan_length = get_conf_val(scan, 'scan_length', 'length')

    if not source is None:
        if args.verbose:
            print("    Applying source configuration...")
        source_location       = get_conf_val(source, 'location_cartesian', 'location')
        source_location_polar = get_conf_val(source, 'location_polar')
        # Default to using Cartesian coordinates
        if source_location is None:
            source_location = shem.geometry.polar2cart(source_location_polar, radians=False)
        source_function_ = get_conf_val(source, 'function')
        # Convert strings representing the source functions to the source functions themselves.
        source_function  = dict(zip((SourceFunctions[key] for key in source_function_.keys()), source_function_.values()))

    if not detector is None:
        if args.verbose:
            print("    Applying detector configuration...")
        detector_location       = get_conf_val(detector, 'location_cartesian', 'location')
        detector_location_polar = get_conf_val(detector, 'location_polar')
        # Default to using Cartesian coordinates
        if detector_location is None:
            detector_location = shem.geometry.polar2cart(detector_location_polar, radians=False)
    
    # Create the mesh using config options.
    if create_mesh:
        if args.verbose:
            print("Creating " + mesh_type + " mesh...")
        
        # Create a flat mesh.
        if mesh_type == 'flat':
            triangles = shem.mesh_manipulation.create_flat_triangles(width, height)
        # Create a mesh with a sphere on the surface.
        elif mesh_type == 'sphere':
            triangles = shem.mesh_manipulation.create_sphere_triangles(width, height, radius)
        # Create a mesh with a rough, randomly generated surface.
        # TODO: Implement rough surface generation.
        elif mesh_type == 'rough':
            triangles = shem.mesh_manipulation.create_flat_triangles(width, height)
        
        # Generate the mesh object.
        mesh = shem.mesh_manipulation.convert_triangles_to_mesh(triangles)

        # Inspect the mesh before using it.
        if inspect_mesh and not args.quiet:
            print("Displaying mesh...")
            # Exit the program if the mesh is not satisfactory.
            if not shem.mesh_manipulation.inspect_mesh(mesh):
                print("Mesh inspection failed. Exiting...")
                sys.exit()
            else:
                print("Mesh inspection complete.")

        # Create any directories required for the mesh file.
        mesh_directory = os.path.split(mesh_file)[0]
        if not os.path.isdir(mesh_directory):
            if not args.quiet:
                print("Creating directory: " + mesh_directory + "...")
            os.makedirs(mesh_directory)
        
        # Save the mesh file, overwriting previous versions
        if not args.quiet:
            if os.path.isfile(mesh_file):
                print("Overwriting " + mesh_file + "...")
            else:
                print("Saving to " + mesh_file + "...")
        shem.mesh_manipulation.save_mesh(mesh, mesh_file)

    # Source Direction
    theta = shem.geometry.cart2polar(source_location)[1]
    phi   = shem.geometry.cart2polar(source_location)[2]
    
    if args.verbose:
        print("Imaging a " + scan_type + "...")

    # Perform the scan simulation
    if scan_type == 'point':
        rays_ = np.stack((
            -shem.source_functions.superposition(ppixels, source_function, theta, phi).reshape((ppixels, 3)),
            np.full((ppixels, 3), source_location),
        ))
        displacement = np.zeros(3)
        rays_[1] += np.tile(displacement, [ppixels, 1]).transpose([0,1])
        detector_location_ = np.tile(displacement, [ppixels, 1]).transpose([0,1]) + detector_location

    elif scan_type == 'line':
        rays_ = np.stack((
            -shem.source_functions.superposition(npixels*ppixels, source_function, theta, phi).reshape((npixels, ppixels, 3)),
            np.full((npixels, ppixels, 3), source_location),
        ))
        displacement = np.empty((npixels, 3))
        x = np.linspace(-scan_length/2, +scan_length/2, npixels)
        displacement[:, 0] = x
        displacement[:, 1] = 0
        displacement[:, 2] = 0
        rays_[1] += np.tile(displacement, [ppixels, 1, 1]).transpose([1,0,2])
        detector_location_ = np.tile(displacement, [ppixels, 1, 1]).transpose([1,0,2]) + detector_location

    elif scan_type == 'plane':
        # Create an the array of rays_ we will be tracing
        rays_ = np.stack((
            # This is negative because the detector's position vector points away from the sample.
            -shem.source_functions.superposition(npixels*npixels*ppixels, source_function, theta, phi).reshape((npixels, npixels, ppixels, 3)),
            np.full((npixels, npixels, ppixels, 3), source_location),
        ))
        displacement = np.empty((npixels, npixels, 3))
        # Create this using meshgrid
        x_ = np.linspace(-scan_length/2, +scan_length/2, npixels)
        x, y = np.meshgrid(x_, x_, sparse=True)
        displacement[:, :, 0] = x
        displacement[:, :, 1] = y
        displacement[:, :, 2] = 0
        # Tile and transpose so that everything adds up.
        rays_[1] += np.tile(displacement, [ppixels, 1, 1, 1]).transpose([1,2,0,3])
        detector_location_ = np.tile(displacement, [ppixels, 1, 1, 1]).transpose([1,2,0,3]) + detector_location

    # Linearise the rays vector for efficiency's sake.
    rays = rays_.reshape(2, -1, 3)
    hits, rays_i, tris_i = shem.ray_tracing.intersects_location(rays, mesh, device, use_torch=False)
    trimesh.points.plot_points(hits)

    # Vector from collision point to detector.
    hits_to_detector = detector_location_.reshape(-1,3)[rays_i] - hits
    hits_to_detector/= np.broadcast_to(np.linalg.norm(hits_to_detector, axis=-1), (3,hits_to_detector.shape[0])).T
    # Need to fudge this so that the ray does not collide with the triangle it just hit.
    dz = 0.00001
    rays_s = np.array([hits_to_detector, hits + np.array([0, 0, dz])])

    # Collision with the mesh before the detector.
    hits_s, rays_i_s_, tris_i_s = shem.ray_tracing.intersects_location(rays_s, mesh, device, use_torch=False)

    # Detected rays not scattered off other surfaces
    rays_i_s = np.delete(rays_i, rays_i_s_)

    print(rays_i.shape, rays_i_s.shape, rays_i_s)
    
    return
