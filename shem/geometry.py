import numpy as np

r, theta, phi = 0, 1, 2
x, y, z = 0, 1, 2

def polar2cart(polar, radians=True):
    '''
    Accept a list of polar coordinates.
    Returns a numpy array containing the corresponding Cartesian coordinates.
    '''
    # Convert between radians and degrees.
    if radians:
        polar_ = polar
    else:
        polar_ = np.array([
            polar[r],
            np.radians(polar[theta]),
            np.radians(polar[phi]),
        ])
    
    # Convert to Cartesians.
    cart = polar_[r] * np.array([ 
        np.cos(polar_[phi]) * np.sin(polar_[theta]),
        np.sin(polar_[phi]) * np.sin(polar_[theta]),
        np.cos(polar_[theta]),
    ])

    return cart

def cart2polar(cart, radians=True):
    '''
    Accept a list of Cartesian coordinates.
    Returns a numpy array containing the corresponding polar coordinates.
    '''

    polar = np.array([
        np.linalg.norm(cart, axis=-1),
        np.arccos(cart[z] / np.linalg.norm(cart, axis=-1)),
        np.arctan2(cart[y], cart[x]),
    ])

    if not radians:
        polar = np.array([
            polar[r],
            np.radians(polar[theta]),
            np.radians(polar[phi]),
        ])

    return polar
