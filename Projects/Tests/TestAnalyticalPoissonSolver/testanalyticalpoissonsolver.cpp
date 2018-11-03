#include <iostream>

#include "Common.h"
#include "Integrator.h"

#include "AnalyticalPoissonSolver.h"

int main(int argc, char** argv)
{
	auto rhs = [](const Vec2R& pos) -> Real
	{
		return 2. * exp(-pos[0] - pos[1]);
	};

	auto solution = [](const Vec2R& pos) -> Real
	{
		return exp(-pos[0] - pos[1]);
	};

	unsigned base = 32;
	unsigned maxbase = base * pow(2,4);
	
	for (; base < maxbase; base *= 2)
	{
		Real dx = M_PI / (Real) base;
		Vec2R origin(0);
		Vec2ui size(round(M_PI / dx));
		Transform xform(dx, origin);

		AnalyticalPoissonSolver solver(xform, size);
		Real error = solver.solve(rhs, solution);

		std::cout << "L-infinity error at " << base << "^2: " << error << std::endl;
	}
}
