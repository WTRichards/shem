{ lib, python3Packages }:
with python3Packages;
buildPythonApplication {
  pname = "shem";
  version = "0.1.0";
  propagatedBuildInputs = [
      numpy
      scipy
      
      cerberus
      pytest
      
      trimesh
      pyglet
      Rtree
      pandas

      matplotlib
      seaborn
      tqdm
  ];
  src = ./.;
}
