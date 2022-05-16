#!/bin/sh

# Run unit tests
# python -m pytest

# For more information about the module
# python -m shem -h

# Generate a mesh comprising a flat platform
#python -m shem -q -m mesh.stl mesh -w 2.0 -T 0.5 -t "flat"

# Generate a mesh comprising several trenches
#python -m shem -q -m mesh.stl mesh -w 2.0 -T 0.5 -W 0.8 -H 0.8 -D 0.2 0.4 0.6 0.8 -t "trenches"

# Generate a mesh comprising a flat platform and sphere
python -m shem -q -m mesh.stl mesh -w 2.0 -T 0.5 -r 0.4 -I 3 -t "sphere"

# Characterise the mesh based on the average solid angle of the rest of the surface visible from any point on the surface.
# The -c 10 excludes the triangles representing the bottom and sides of the sample
#python -m shem -m mesh.stl -g characterise -b 128 -n 10000 -c 10

# For more information about the mesh subcommand
# python -m shem mesh -h

# Copy default_config.py to config.py in the work directory if config.py does not exist, otherwise run the simulation using config.py on the CPU.
#python -m shem -v generate config.py

# Run the simulation and generate an image using config.py on the GPU.
python -O -m shem -m mesh.stl -g -v generate config.py

# For more information about the generate subcommand
# python -m shem generate -h

# Run the analysis program using the image generated by generate above.
#python -O -m shem -g -v analyse config.py example_image.png

# For more information about the analyse subcommand
# python -m shem generate -h
