#include <memory>

#include "core.h"

#include "Transform.h"

#include "AnalyticalViscositySolver.h"

static std::unique_ptr<AnalyticalViscositySolver> g_solver;

int main(int argc, char** argv)
{

	Real dt = 1.; Real mu = .1;
	// TODO: add variable viscosity
	auto initial = [&](const Vec2R& pos)
	{
		return (1. + 3. * dt * mu) * sin(pos[0]) * sin(pos[1]) - dt * mu * cos(pos[0]) * cos(pos[1]);
	};

	auto solution = [](const Vec2R& pos) { return sin(pos[0]) * sin(pos[1]); };

	int base = 16;
	int maxbase = base * pow(2,4);
	for (; base < maxbase; base *= 2)
	{
		Real dx = M_PI / (Real)base;
		Vec2R origin(0);
		Vec2st size(round(M_PI / dx));
		Transform xform(dx, origin);

		g_solver = std::make_unique<AnalyticalViscositySolver>(xform, size);
		Real error = g_solver->solve(initial, solution, dt, mu, 0);

		std::cout << "L-infinity error at " << base << "^2: " << error << std::endl;
	}
}