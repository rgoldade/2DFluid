# 2dFluidSimulation
This is a 2D fluid simulation project based largley on the material found in Robert Bridson's textbook "Fluid Simulation for Computer Graphics, 2nd Edition". The code contains implementations for the staggered-grid Eulerian pressure project and viscosity solve, as well as surface tracking methods for level sets, explicity simplicial meshes and fluid particles (markers, PIC, FLIP).

To build the project in Linux, create a new folder in the root directly (usually "build"), type cmake .. then make. The binaries to run the simulations should compile in the ./bin folder. The project requires Eigen3 and glut to be installed and findable.

For VS, use the cmake-gui to generate the solution file. Make sure to include the path to your Eigen3 and freeglut folders.

To-do:

1. Add FLIP simulation scene (mechanics are present but no loop has be added to this repo).
2. Make sure reseeding is deterministic (use a seed based on cell position, etc.,). This is important for repeatability.
3. Add TBB parallelism to embarassingly parallel loops (e.g., labelling active cells for pressure projection or active faces for viscosity).
4. Add fast iterative method to replace fast marching method for rebuilding level set (to be parallel and relatively cache efficient).
5. Add a scene with moving collisions to demonstrate that it indeed works.
6. Add support and a scene for variable viscosity.
7. Test build with Mac.

Stretch to-do:

1. Use pybind, swig or equivalent to script scene files and simulation loops in python.
2. APIC velocity transfer.
