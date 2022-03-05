import pytest

if __name__ == '__main__':
    print("Running unit tests...")


def func(x):
    return x+1

def test_answer():
    assert func(3) == 5

