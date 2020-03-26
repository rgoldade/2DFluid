#include <iostream>
#include <memory>

#include "AnalyticalViscositySolver.h"
#include "Transform.h"
#include "Utilities.h"

int main(int argc, char** argv)
{
	float dt = 1., mu = .1;

	auto initial = [&](const Vec2f& pos, unsigned axis)
	{
		float x = pos[0];
		float y = pos[1];
		float val;
		if (axis == 0)
			val = sin(x) * sin(y) - dt * (2. / PI * cos(x) * sin(y) + (cos(x + y) - 2 * sin(x) * sin(y))*(x / PI + .5));
		else
			val = sin(x) * sin(y) - dt * ((cos(x) * cos(y) - 3. * sin(x) * sin(y)) * (x / PI + .5) + 1. / PI * sin(x + y));

		return val;
	};

	auto solution = [](const Vec2f& pos, unsigned axis) { return sin(pos[0]) * sin(pos[1]); };
	auto viscosity = [](const Vec2f& pos) { return pos[0] / PI + .5; };

	unsigned baseGrid = 16;
	unsigned maxBaseGrid = baseGrid * pow(2,4);

	for (; baseGrid < maxBaseGrid; baseGrid *= 2)
	{
		float dx = PI / float(baseGrid);
		Vec2f origin(0);
		Vec2i size(round(PI / dx));
		Transform xform(dx, origin);

		AnalyticalViscositySolver solver(xform, size);
		float error = solver.solve(initial, solution, viscosity, dt);

		std::cout << "L-infinity error at " << baseGrid << "^2: " << error << std::endl;
	}
}
