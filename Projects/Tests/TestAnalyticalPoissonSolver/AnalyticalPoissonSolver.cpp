#include "AnalyticalPoissonSolver.h"

#include "Renderer.h"

void AnalyticalPoissonSolver::drawGrid(Renderer &renderer) const
{
	myPoissonGrid.drawGrid(renderer);
}

void AnalyticalPoissonSolver::drawValues(Renderer &renderer) const
{
	myPoissonGrid.drawSupersampledValues(renderer);
}