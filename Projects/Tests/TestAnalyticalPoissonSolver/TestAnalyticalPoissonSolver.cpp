#include <iostream>

#include "AnalyticalPoissonSolver.h"

#include "Common.h"
#include "Integrator.h"

int main(int argc, char** argv)
{
	auto rhs = [](const Vec2R& pos) -> Real
	{
		return 2. * std::exp(-pos[0] - pos[1]);
	};

	auto solution = [](const Vec2R& pos) -> Real
	{
		return std::exp(-pos[0] - pos[1]);
	};

	unsigned baseGrid = 32;
	unsigned maxBaseGrid = baseGrid * pow(2,4);
	
	for (; baseGrid < maxBaseGrid; baseGrid *= 2)
	{
		Real dx = Util::PI / Real(baseGrid);
		Vec2R origin(0);
		Vec2ui size(round(Util::PI / dx));
		Transform xform(dx, origin);

		AnalyticalPoissonSolver solver(xform, size);
		Real error = solver.solve(rhs, solution);

		std::cout << "L-infinity error at " << baseGrid << "^2: " << error << std::endl;
	}
}
