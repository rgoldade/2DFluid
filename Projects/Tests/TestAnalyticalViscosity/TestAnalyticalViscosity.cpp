#include <iostream>
#include <memory>

#include "AnalyticalViscositySolver.h"
#include "Transform.h"
#include "Utilities.h"

using namespace FluidSim2D;

int main(int, char**)
{
	double dt = 1.;

	auto initial = [&](const Vec2d& pos, unsigned axis)
	{
		double x = pos[0];
		double y = pos[1];
		double val;
		if (axis == 0)
			val = sin(x) * sin(y) - dt * (2. / PI * cos(x) * sin(y) + (cos(x + y) - 2 * sin(x) * sin(y))*(x / PI + .5));
		else
			val = sin(x) * sin(y) - dt * ((cos(x) * cos(y) - 3. * sin(x) * sin(y)) * (x / PI + .5) + 1. / PI * sin(x + y));

		return val;
	};

	auto solution = [](const Vec2d& pos, int) { return sin(pos[0]) * sin(pos[1]); };
	auto viscosity = [](const Vec2d& pos) { return pos[0] / PI + .5; };

	const int startGrid = 16;
	const int endGrid = startGrid * int(pow(2,4));

	for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
	{
		const double dx = PI / double(gridSize);
		Vec2d origin = Vec2d::Zero();
		Vec2i size(round(PI / dx), round(PI / dx));
		Transform xform(dx, origin);

		AnalyticalViscositySolver solver(xform, size);
		double error = solver.solve(initial, solution, viscosity, dt);

		std::cout << "L-infinity error at " << gridSize << "^2: " << error << std::endl;
	}
}
