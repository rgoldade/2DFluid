# 2dFluid
This is a 2D fluid simulation project based largley on the material found in Robert Bridson's textbook "Fluid Simulation for Computer Graphics, 2nd Edition". The code contains implementations for the staggered-grid Eulerian pressure projection and viscosity solve, as well as surface tracking methods for level sets, explicit simplicial meshes and fluid particles (markers, PIC, FLIP).

To build the project in Linux, create a new folder in the root directly (usually "build"), type cmake .. then make. The binaries to run the simulations should compile in the ./bin folder. The project requires Eigen, tbb, and glut to be installed and findable.

For VS, use the cmake-gui to generate the solution file. Make sure to include the path to your Eigen3, freeglut, and tbb folders.

Legal stuff:

Copyright 2018 Ryan Goldade

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
