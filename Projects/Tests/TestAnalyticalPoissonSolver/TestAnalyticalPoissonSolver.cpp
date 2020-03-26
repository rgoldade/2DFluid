#include <iostream>

#include "AnalyticalPoissonSolver.h"

#include "Integrator.h"
#include "Utilities.h"


int main(int argc, char** argv)
{
	auto rhs = [](const Vec2f& pos) -> float
	{
		return 2. * std::exp(-pos[0] - pos[1]);
	};

	auto solution = [](const Vec2f& pos) -> float
	{
		return std::exp(-pos[0] - pos[1]);
	};

	int baseGrid = 32;
	int maxBaseGrid = baseGrid * pow(2,4);
	
	for (; baseGrid < maxBaseGrid; baseGrid *= 2)
	{
		float dx = PI / float(baseGrid);
		Vec2f origin(0);
		Vec2i size(round(PI / dx));
		Transform xform(dx, origin);

		AnalyticalPoissonSolver solver(xform, size);
		float error = solver.solve(rhs, solution);

		std::cout << "L-infinity error at " << baseGrid << "^2: " << error << std::endl;
	}
}
