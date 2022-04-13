import sys

import pytest

import numpy as np
import shem.geometry
from shem.definitions import *

if __name__ == '__main__':
    print("Running unit tests...")


x_cart = np.array([1,0,0])
y_cart = np.array([0,1,0])
z_cart = np.array([0,0,1])

x_polar_deg = np.array([1,0,90])
y_polar_deg = np.array([1,90,90])
z_polar_deg = np.array([1,0,0])

def is_close(a, b):
    return np.allclose(a, b, atol=5e-6)

############################
# shem.geometry.polar2cart #
############################

def _polar2cart_xyz():
    assert is_close(shem.geometry.polar2cart(x_polar_deg, radians=False), x_cart)
    assert is_close(shem.geometry.polar2cart(y_polar_deg, radians=False), y_cart)
    assert is_close(shem.geometry.polar2cart(z_polar_deg, radians=False), z_cart)
    
@pytest.mark.fast
def test_polar2cart():
    # Chech polar2cart works for unit x, y and z vectors
    _polar2cart_xyz()

############################
# shem.geometry.cart2polar #
############################

def _cart2polar_xyz():
    assert is_close(shem.geometry.cart2polar(x_cart, radians=False), x_polar_deg)
    assert is_close(shem.geometry.cart2polar(y_cart, radians=False), y_polar_deg)
    assert is_close(shem.geometry.cart2polar(z_cart, radians=False), z_polar_deg)
    
@pytest.mark.fast
def test_cart2polar():
    # Chech cart2polar works for unit x, y and z vectors
    _cart2polar_xyz()

############################
# shem.geometry.vector_dot #
############################

def _vector_dot_basis_vectors():
    x, y, z = x_cart, y_cart, z_cart
    assert shem.geometry.vector_dot(x, x) == 1
    assert shem.geometry.vector_dot(x, y) == 0
    assert shem.geometry.vector_dot(x, z) == 0
    assert shem.geometry.vector_dot(y, x) == 0
    assert shem.geometry.vector_dot(y, y) == 1
    assert shem.geometry.vector_dot(y, z) == 0
    assert shem.geometry.vector_dot(z, x) == 0
    assert shem.geometry.vector_dot(z, y) == 0
    assert shem.geometry.vector_dot(z, z) == 1

def _vector_dot_commutative():
    x, y, z = x_cart, y_cart, z_cart
    np.random.seed(0)
    v0 = np.random.rand(3)
    v1 = np.random.rand(3)
    assert shem.geometry.vector_dot(v0, v1) == shem.geometry.vector_dot(v1, v0)

def _vector_dot_broadcast():
    x, y, z = x_cart, y_cart, z_cart
    np.random.seed(0)
    v0 = np.random.rand(3)
    assert is_close(shem.geometry.vector_dot(v0, np.array([x,y,z])), v0)


def test_vector_dot():
    # Check we get the results we expect for the Cartesian basis vectors.
    _vector_dot_basis_vectors()
    # Check the dot product is commutative for arbitrary vectors.
    _vector_dot_commutative()
    # Check this dot product works if the sets of vectors supplied differ in shape.
    _vector_dot_broadcast()

#############################
# shem.geometry.vector_norm #
#############################

def _vector_norm_known():
    v = np.array([
        # Pythagorean triples
        [0, 3, 4],
        [5, 0, 12],
        # Pythagorean quadruples
        [1, 2, 2],
        [2, 1, 2],
        [4, 4, 7]
    ])

    expected = np.array([
        5,
        13,
        3,
        3,
        9
    ])

    assert is_close( shem.geometry.vector_norm(v), expected )

def _vector_norm_and_dot_product():
    n = 256
    d = 3
    np.random.seed(0)
    v = np.random.rand(n, d)
    assert is_close( np.sqrt(shem.geometry.vector_dot(v, v)), shem.geometry.vector_norm(v))

def test_vector_norm():
    # Check that the vector norm is equivalent to the square root of the dot product.
    _vector_norm_and_dot_product()
    # Check that the function works on known solutions
    _vector_norm_known()

##############################
# shem.geometry.vector_cross #
##############################

def _vector_cross_anticommutative():
    n = 256
    d = 3
    np.random.seed(0)
    v0 = np.random.rand(n, d)
    v1 = np.random.rand(n, d)
    assert is_close( shem.geometry.vector_cross(v0, v1), -shem.geometry.vector_cross(v1, v0) )

def _vector_cross_known():
    v0 = np.array([
        x_cart,
        [1, 2, 3]
    ])

    v1 = np.array([
        y_cart,
        [2, 4, 1],
    ])

    expected = np.array([
        z_cart,
        [-10, 5, 0],
    ])

def test_vector_cross():
    # Check that the cross product is anticommutative
    _vector_cross_anticommutative()
    # Check that the function works on known solutions
    _vector_cross_known()

##################################
# shem.geometry.vector_normalise #
##################################

def _vector_normalise_basis_vectors():
    x, y, z = x_cart, y_cart, z_cart
    assert is_close(shem.geometry.vector_normalise(np.array([x,y,z])), np.array([x,y,z]))

def test_vector_normalise():
    # Test basis vectors
    _vector_normalise_basis_vectors()

#############################
# shem.geometry.unit_vector #
#############################

def _unit_vector_basis_vectors():
    x, y, z = x_cart, y_cart, z_cart
    # Angular components
    basis_vectors_polar = np.array([x_polar_deg, y_polar_deg, z_polar_deg])[..., 1:]
    assert is_close( shem.geometry.unit_vector(basis_vectors_polar, radians=False), np.array([x,y,z]) )

def test_unit_vector():
    # Check we can recover the Cartesian basis vectors from their angular components.
    _unit_vector_basis_vectors()

##############################
# shem.geometry.rotate_frame #
##############################

def _rotate_frame_cart():
    x, y, z = x_cart, y_cart, z_cart
    cart = np.array([x,y,z])
    assert is_close( shem.geometry.rotate_frame(x, x, cart), cart )
    assert is_close( shem.geometry.rotate_frame(x, y, cart), np.array([+y, -x, +z]) )
    assert is_close( shem.geometry.rotate_frame(x, z, cart), np.array([+z, +y, -x]) )
    assert is_close( shem.geometry.rotate_frame(y, x, cart), np.array([-y, +x, +z]) )
    assert is_close( shem.geometry.rotate_frame(y, y, cart), cart )
    assert is_close( shem.geometry.rotate_frame(y, z, cart), np.array([+x, +z, -y]) )
    assert is_close( shem.geometry.rotate_frame(z, x, cart), np.array([-z, +y, +x]) )
    assert is_close( shem.geometry.rotate_frame(z, y, cart), np.array([+x, -z, +y]) )
    assert is_close( shem.geometry.rotate_frame(z, z, cart), cart )

def _rotate_frame_inverse():
    n = 256
    np.random.seed(0)

    arr = np.random.randn(n,3)
    v_i = np.random.randn(n,3)
    v_f = np.random.randn(n,3)

    for i in range(n):
        rotated      =  shem.geometry.rotate_frame(v_i[i], v_f[i], arr)
        rotated_back =  shem.geometry.rotate_frame(v_f[i], v_i[i], rotated)
        assert is_close( arr, rotated_back)

def _rotate_frame_preserved_dot_product():
    n = 256
    np.random.seed(0)

    arr = np.random.randn(n,3)
    v_i = shem.geometry.vector_normalise(np.random.randn(n,3))
    v_f = shem.geometry.vector_normalise(np.random.randn(n,3))

    for i in range(n):
        rotated      = shem.geometry.rotate_frame(v_i[i], v_f[i], arr)
        rotated_back = shem.geometry.rotate_frame(v_f[i], v_i[i], rotated)

        old_dot = shem.geometry.vector_dot(    arr, v_i[i] )
        new_dot = shem.geometry.vector_dot(rotated, v_f[i] )
        assert is_close( old_dot, new_dot )

def test_rotate_frame():
    # Check the function works for the Cartesian basis vectors.
    _rotate_frame_cart()
    # Check that we can invert the function by swapping intial and final vectors.
    _rotate_frame_inverse()
    # Check that we preserve the angle between the rotated vector and the new axis.
    _rotate_frame_preserved_dot_product()

##############################
# shem.geometry.vector_angle #
##############################

def _vector_angle_cart():
    x, y, z = x_cart, y_cart, z_cart
    cart = np.array([x,y,z])

    assert is_close( shem.geometry.vector_angle(x, cart, radians=False), np.array([00, 90, 90]) )
    assert is_close( shem.geometry.vector_angle(y, cart, radians=False), np.array([90, 00, 90]) )
    assert is_close( shem.geometry.vector_angle(z, cart, radians=False), np.array([90, 90, 00]) )


def _vector_angle_known():
    a = np.array([
        [1,1,0],
        [1,1,0],
    ])
    
    b = np.array([
        [1,1,0],
        [1,0,0],
    ])

    result = np.array([
        0,
        45,
    ])

    print(shem.geometry.vector_angle(a, b, radians=False))

    assert is_close( shem.geometry.vector_angle(a, b, radians=False), result )


def test_vector_angle():
    # Check the function works for the Cartesian basis vectors.
    _vector_angle_cart()
    # Check the function works for some known values
    _vector_angle_known()

