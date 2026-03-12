— Race condition in SparseTile::setVoxel (SparseUniformGrid.h:70) — Two threads calling setVoxel on a constant tile can both trigger expand() simultaneously. Already has a // TODO: add lock comment in the code.

— Full grid copy every Jacobi smoother iteration (GeometricMultigridOperators.cpp:115) — UniformGrid<double> tempSolution = solution; copies the entire grid every call. Could accept a pre-allocated scratch buffer or switch to red-black Gauss-Seidel.

— Initial guess permanently disabled (MultiMaterialPressureProjection.cpp:698) — if (false/*myUseInitialGuessPressure*/) means setInitialGuess() has no effect on MultiMaterialPressureProjection.

— Dead member variable myNewParticles (FluidParticles.h:83) — Populated in reseed() but never read.

— No CG breakdown guard (GeometricConjugateGradientSolver.h:77,99) — alpha = absNew / dotProduct(...) and beta = absNew / absOld have no zero-denominator checks, risking division by zero or infinity on CG breakdown.