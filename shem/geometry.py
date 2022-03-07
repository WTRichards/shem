import numpy as np

r, theta, phi = 0, 1, 2
x, y, z = 0, 1, 2

def polar2cart(polar_, radians=True):
    '''
    Accept a list of polar coordinates.
    Returns a numpy array containing the corresponding Cartesian coordinates.
    '''
    # Convert between radians and degrees.
    polar = np.array(polar_, dtype=np.float32)
    cart = np.empty_like(polar)

    if not radians:
        polar[..., theta] = np.radians(polar[..., theta])
        polar[..., phi] = np.radians(polar[..., phi])
    
    # Convert to Cartesians.
    cart[..., x] = polar[..., r] * np.sin(polar[..., phi]) * np.cos(polar[..., theta])
    cart[..., y] = polar[..., r] * np.sin(polar[..., phi]) * np.sin(polar[..., theta])
    cart[..., z] = polar[..., r] * np.cos(polar[..., phi])


    return cart

def cart2polar(cart_, radians=True):
    '''
    Accept a list of Cartesian coordinates.
    Returns a numpy array containing the corresponding polar coordinates.
    '''
    cart = np.array(cart_, dtype=np.float32)
    polar = np.empty_like(cart)

    polar[..., r] = np.linalg.norm(cart, axis=-1)
    polar[..., theta] = np.arctan2(cart[..., y], cart[..., x])
    polar[..., phi] = np.arccos(cart[..., z] / polar[..., r])

    if not radians:
        polar[..., theta] = np.degrees(polar[..., theta])
        polar[..., phi] = np.degrees(polar[..., phi])

    return polar
