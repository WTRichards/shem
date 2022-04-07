import sys

import pytest

import numpy as np
import shem.source
import shem.default_config
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
    return np.allclose(a, b, atol=DELTA)

#########################
# shem.source.direction #
#########################

def test_direction():
    n_rays = 256
    a = np.empty((n_rays, 256), dtype=np.float32)
    source_angle = np.radians(1)


