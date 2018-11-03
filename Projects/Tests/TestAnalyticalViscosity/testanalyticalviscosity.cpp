#include <memory>
#include <iostream>

#include "Common.h"
#include "Transform.h"
#include "AnalyticalViscositySolver.h"

int main(int argc, char** argv)
{
	Real dt = 1., mu = .1;

	auto initial = [&](const Vec2R& pos, unsigned axis)
	{
		Real x = pos[0];
		Real y = pos[1];
		Real val;
		if (axis == 0)
			val = sin(x) * sin(y) - dt * (2. / M_PI * cos(x) * sin(y) + (cos(x + y) - 2 * sin(x)*sin(y))*(x / M_PI + .5));
		else
			val = sin(x) * sin(y) - dt * ((cos(x) * cos(y) - 3. * sin(x) * sin(y)) * (x / M_PI + .5) + 1. / M_PI * sin(x + y));

		return val;
	};

	auto solution = [](const Vec2R& pos, unsigned axis) { return sin(pos[0]) * sin(pos[1]); };
	auto viscosity = [](const Vec2R& pos) { return pos[0] / M_PI + .5; };

	unsigned base = 16;
	unsigned maxbase = base * pow(2,4);
	for (; base < maxbase; base *= 2)
	{
		Real dx = M_PI / (Real)base;
		Vec2R origin(0);
		Vec2ui size(round(M_PI / dx));
		Transform xform(dx, origin);

		AnalyticalViscositySolver solver(xform, size);
		Real error = solver.solve(initial, solution, viscosity, dt);

		std::cout << "L-infinity error at " << base << "^2: " << error << std::endl;
	}
}
