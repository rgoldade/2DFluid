#include <iostream>
#include <memory>

#include "AnalyticalViscositySolver.h"
#include "Common.h"
#include "Transform.h"

int main(int argc, char** argv)
{
	Real dt = 1., mu = .1;

	auto initial = [&](const Vec2R& pos, unsigned axis)
	{
		Real x = pos[0];
		Real y = pos[1];
		Real val;
		if (axis == 0)
			val = sin(x) * sin(y) - dt * (2. / Util::PI * cos(x) * sin(y) + (cos(x + y) - 2 * sin(x)*sin(y))*(x / Util::PI + .5));
		else
			val = sin(x) * sin(y) - dt * ((cos(x) * cos(y) - 3. * sin(x) * sin(y)) * (x / Util::PI + .5) + 1. / Util::PI * sin(x + y));

		return val;
	};

	auto solution = [](const Vec2R& pos, unsigned axis) { return sin(pos[0]) * sin(pos[1]); };
	auto viscosity = [](const Vec2R& pos) { return pos[0] / Util::PI + .5; };

	unsigned baseGrid = 16;
	unsigned maxBaseGrid = baseGrid * pow(2,4);

	for (; baseGrid < maxBaseGrid; baseGrid *= 2)
	{
		Real dx = Util::PI / Real(baseGrid);
		Vec2R origin(0);
		Vec2i size(round(Util::PI / dx));
		Transform xform(dx, origin);

		AnalyticalViscositySolver solver(xform, size);
		Real error = solver.solve(initial, solution, viscosity, dt);

		std::cout << "L-infinity error at " << baseGrid << "^2: " << error << std::endl;
	}
}
