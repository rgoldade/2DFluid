#include "FlipSimulation.h"

void FlipSimulation::draw_grid(Renderer& renderer) const
{
	m_surface.draw_grid(renderer);
}

void FlipSimulation::draw_surface(Renderer& renderer)
{
	m_surface.draw_surface(renderer, Vec3f(0., 0., 1.0));
	m_particles.draw_points(rendererVec3f(0., 0., 1.0));
}

void FlipSimulation::draw_collision(Renderer& renderer)
{
	m_collision.draw_surface(renderer);
}

void FlipSimulation::draw_velocity(Renderer& renderer, Real length) const
{
	m_particles.draw_velocity(renderer, length);
}

