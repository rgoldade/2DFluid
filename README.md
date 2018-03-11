# 2dFluidSimulation
This is a 2D fluid simulation project based largley on the material found in Robert Bridson's textbook "Fluid Simulation for Computer Graphics, 2nd Edition". The code contains implementations for the staggered-grid Eulerian pressure projection and viscosity solve, as well as surface tracking methods for level sets, explicit simplicial meshes and fluid particles (markers, PIC, FLIP).

To build the project in Linux, create a new folder in the root directly (usually "build"), type cmake .. then make. The binaries to run the simulations should compile in the ./bin folder. The project requires Eigen3 and glut to be installed and findable.

For VS, use the cmake-gui to generate the solution file. Make sure to include the path to your Eigen3, freeglut, and tbb folders.

To-do:

1. Add FLIP simulation scene (mechanics are present but no loop has be added to this repo).
2. Make sure reseeding is deterministic (use a seed based on cell position, etc.,). This is important for repeatability.
3. Add TBB parallelism to embarassingly parallel loops (e.g., labelling active cells for pressure projection or active faces for viscosity).
4. Add fast iterative method to replace fast marching method for rebuilding level set (to be parallel and relatively cache efficient).
5. Add a scene with moving collisions to demonstrate that it indeed works.
6. Add support and a scene for variable viscosity.
7. Test build with Mac.
8. Consistency when indexing loops -- (i,j) or (x,y)

Stretch to-do:

1. Use pybind, swig or equivalent to script scene files and simulation loops in python.
2. APIC velocity transfer.

Legal stuff:

Copyright 2018 Ryan Goldade

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
