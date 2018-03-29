#include "Core.h"
#include "Vec.h"
#include "Integrator.h"

#include "AnalyticalPoissonSolver.h"

int main(int argc, char** argv)
{
	auto initial = [&](const Vec2R& pos) -> Real
	{
		2. * exp(-pos[0] - pos[1]);
	};

	auto solution = [&](const Vec2R& pos) -> Real
	{
		exp(-pos[0] - pos[1]);
	};

	int base = 32;
	int maxbase = base * pow(2,4);
	
	for (; base < maxbase; base *= 2)
	{
		Real dx = M_PI / (Real) base;
		Vec2R origin(0);
		Vec2st size(round(M_PI / dx));
		Transform xform(dx, origin);

		AnalyticalPoissonSolver solver(xform, size);
		Real error = solver.solve(initial, solution);

		std::cout << "L-infinity error at " << base << "^2: " << error << std::endl;
	}
}
