{ lib, python3Packages }:
with python3Packages;
buildPythonApplication {
  pname = "shem";
  version = "0.1.0";
  propagatedBuildInputs = [
      cerberus
      matplotlib
      numpy
      pandas
      pyglet
      pytest
      pytorch
      Rtree
      scipy
      tqdm
      trimesh
      numba
      pyopencl
      seaborn
      dash
  ];
  src = ./.;
}
