#!/bin/sh

# Run unit tests
# python -m pytest

# For more information about the module
# python -m shem -h

# Generate a mesh comprising a flat platform
python -m shem -q -m mesh.stl mesh -W 1.0 -H 0.2 -t "flat"

# Generate a mesh comprising a flat platform and sphere
#python -m shem -q -m mesh.stl mesh -W 1.0 -H 0.2 -R 0.1 -I 3 -t "sphere"

# For more information about the mesh subcommand
# python -m shem mesh -h

# Copy default_config.py to config.py in the work directory.
python -m shem -v generate config.py

# Run the simulation using config.py on the CPU.
python -m shem -v generate config.py
# Run the simulation using config.py on the GPU.
#python -O -m shem -g -v generate config.py

# For more information about the generate subcommand
# python -m shem generate -h

