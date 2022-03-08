import toml
import os

import numpy as np
import trimesh
import tqdm

import shem.display
import shem.geometry
import shem.mesh_manipulation
import shem.scattering_functions
import shem.source_functions

ScanTypes = {'point', 'line', 'plane'}
MeshTypes  = {'flat', 'sphere', 'rough'}
SourceFunctions = {
        'delta': shem.source_functions.delta,
        'uniform_cone': shem.source_functions.uniform_cone,
}
ScatteringFunctions = {
        'specular': shem.scattering_functions.specular,
        'diffuse': shem.scattering_functions.diffuse,
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

def run_config(args, config_file):
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
        dpixels  = get_conf_val(sim, 'dpixels')
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

    # Need a dictionary indexed by functions with kwargs keys.
    if not scat is None:
        if args.verbose:
            print("    Applying scattering distribution configuration...")
        scat_normalise = get_conf_val(scat, 'normalise', 'monte_carlo_normalise')
        brute_force = get_conf_val(scat, 'brute_force')
        scattering_function_ = get_conf_val(scat, 'function')
        # Convert strings representing the source functions to the source functions themselves.
        if not scattering_function_ is None:
            scattering_function  = dict(zip((ScatteringFunctions[key] for key in scattering_function_.keys()), scattering_function_.values()))

    if not scan is None:
        if args.verbose:
            print("    Applying scan configuration...")
        scan_type   = get_conf_val(scan, 'scan_type', 'type')
        scan_length = get_conf_val(scan, 'scan_length', 'length')
        inspect_scan = get_conf_val(scan, 'inspect', 'inspect_model', 'inspect_scan')

    if not source is None:
        if args.verbose:
            print("    Applying source configuration...")
        source_location       = get_conf_val(source, 'location_cartesian', 'location')
        source_location_polar = get_conf_val(source, 'location_polar')
        # Default to using Cartesian coordinates
        if source_location is None:
            source_location = shem.geometry.polar2cart(source_location_polar, radians=False)
        source_normalise = get_conf_val(source, 'normalise', 'monte_carlo_normalise')
        source_radius = get_conf_val(source, 'source_radius', 'radius')
        source_function_ = get_conf_val(source, 'function')
        if not source_function_ is None:
            source_function  = dict(zip((SourceFunctions[key] for key in source_function_.keys()), source_function_.values()))

    if not detector is None:
        if args.verbose:
            print("    Applying detector configuration...")
        detector_location       = get_conf_val(detector, 'location_cartesian', 'location')
        detector_location_polar = get_conf_val(detector, 'location_polar')
        # Default to using Cartesian coordinates
        if detector_location is None:
            detector_location = shem.geometry.polar2cart(detector_location_polar, radians=False)
        detector_radius = get_conf_val(detector, 'detector_radius', 'radius')
   
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
            print("Displaying mesh, source and detector...")
            scene = trimesh.scene.scene.Scene(mesh)
            scene.show()
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

    if args.verbose:
        print("Imaging a " + scan_type + "...")

    # Define the parameters of the scan.
    if scan_type == 'point':
        x_dim, y_dim = 1, 1
        x = np.array([0.0])
        y = np.array([0.0])

    elif scan_type == 'line':
        x_dim, y_dim = npixels, 1
        x = np.linspace(-scan_length/2, +scan_length/2, x_dim)
        y = np.array([0.0])

    elif scan_type == 'plane':
        x_dim, y_dim = npixels, npixels
        x = np.linspace(-scan_length/2, +scan_length/2, x_dim)
        y = np.linspace(-scan_length/2, +scan_length/2, y_dim)

    else:
        raise ValueError("Invalid scan_type: " + scan_type)

    # Source Direction
    theta = shem.geometry.cart2polar(source_location, radians=True)[1]
    phi   = shem.geometry.cart2polar(source_location, radians=True)[2]
    
    # Create the initial set of rays.
    rays_ = np.stack((
        shem.source_functions.superposition(x_dim*y_dim*ppixels, source_function, theta, phi).reshape((x_dim, y_dim, ppixels, 3)),
        np.full((x_dim, y_dim, ppixels, 3), source_location),
    ))

    # Create the displacement vectors corresponding to the scan.
    displacement_ = np.empty((x_dim, y_dim, 3))
    yv, xv = np.meshgrid(x, y, sparse=True)
    displacement_[:, :, 0] = xv
    displacement_[:, :, 1] = yv
    displacement_[:, :, 2] = 0
    # Adjust based on the center of mass of the mesh.
    displacement_ += mesh.center_mass

    pixels = np.zeros((x_dim, y_dim), dtype=np.float32)

    # Inspect the mesh before using it.
    if inspect_scan and not args.quiet:
        print("Displaying intended scanning procedure...")
        scene = trimesh.scene.scene.Scene(mesh)
        scene.add_geometry(trimesh.points.PointCloud(source_location - displacement_.reshape(-1,3)))
        scene.add_geometry(trimesh.points.PointCloud(detector_location - displacement_.reshape(-1,3)))
        # Plot the rays
        for t in np.linspace(0,1,11):
            scene.add_geometry(trimesh.points.PointCloud((rays_[0]*t + rays_[1] - displacement_.reshape(x_dim, y_dim, 1, 3)).reshape(-1,3)))
        scene.show()
        print("Model inspection complete.")

    
    index_loop = tqdm.trange(x_dim*y_dim)
    index_loop.set_description("Running " + config_file)
    # Enumerate over the scanning displacement.
    for index in np.ndindex(x_dim, y_dim):
        i = index[0]
        j = index[1]
        # Apply the displacement vectors to the rays and detector location.
        rays = rays_[:, i, j, :, :]
        displacement = displacement_[index]
        rays[1] -= displacement
        source_loc = source_location - displacement
        # Sample multiple points within the detector radius.
        if dpixels == 0:
            detector_loc = (detector_location - displacement).reshape(1,-1)
        else:
            detector_loc = detector_location - displacement + detector_radius*shem.geometry.polar2cart(np.array([
                np.ones(dpixels),
                2*np.pi*np.random.rand(dpixels),
                np.pi*np.random.rand(dpixels),
            ]).T)

        # Calculate the hit location, the corresponding ray and the corresponding triangle of each hit.
        hits, rays_i, tris_i = shem.ray_tracing.intersects_location(rays, mesh, args.use_gpu)
        if rays_i.size != 0:
            for det_loc in detector_loc:
                # Calculate the vector from the hit location to the detector and normalise.
                V_ = det_loc - hits
                V_/= np.linalg.norm(V_, axis=-1).reshape(-1,1)
            
                # Vector from collision point to detector.
                # Need to fudge this so that the ray does not collide with the triangle it just hit.
                # This is done by adding a small amount times the normal vector.
                # We can't use the smallest possible float since this is multiplied element-wise, so would just become zero.
                delta = 0.0001
                rays_s = np.array([V_, hits + delta*mesh.face_normals[tris_i]])
            
                # Rays which collide with the mesh before the detector.
                collisions = shem.ray_tracing.intersects_any(rays_s, mesh, args.use_gpu)
                hits_detected = hits[np.logical_not(collisions)]
                rays_i_detected = rays_i[np.logical_not(collisions)]
                tris_i_detected = tris_i[np.logical_not(collisions)]
            
                # Check if any rays are detected.
                if collisions.size != 0:
                    # Vector to detector.
                    V = det_loc - hits_detected
                    V/= np.linalg.norm(V, axis=-1).reshape(-1,1)
                    # Vector to source.
                    L = source_loc - hits_detected
                    L/= np.linalg.norm(L, axis=-1).reshape(-1,1)
                    # Normals of each triangle.
                    N = mesh.face_normals[tris_i_detected]
                    N/= np.linalg.norm(N, axis=-1).reshape(-1,1)
                    # Reflected vectors.
                    R = L - 2*N*(L*N).sum(-1).reshape(-1,1)
    
                    # Calculation of pixel value based on scattering
                    pixels[index] += shem.scattering_functions.superposition(scattering_function, L, N, -R, V)
        index_loop.update()
    index_loop.close()
    if dpixels != 0:
        pixels /= dpixels
    
    shem.display.save_image(pixels, config_file+'.png')
    
    return
