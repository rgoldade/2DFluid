#include "AnalyticalPoissonSolver.h"
#include "Renderer.h"

void AnalyticalPoissonSolver::draw_grid(Renderer &renderer) const
{
	m_poissongrid.draw_grid(renderer);
}

void AnalyticalPoissonSolver::draw_values(Renderer &renderer) const
{
	m_poissongrid.draw_supersampled_values(renderer);
}