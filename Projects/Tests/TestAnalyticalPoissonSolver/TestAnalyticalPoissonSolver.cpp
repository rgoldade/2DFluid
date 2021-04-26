#include <iostream>

#include "AnalyticalPoissonSolver.h"

#include "Integrator.h"
#include "Utilities.h"


int main(int, char**)
{
	auto rhs = [](const Vec2d& pos) -> double
	{
		return 2. * std::exp(-pos[0] - pos[1]);
	};

	auto solution = [](const Vec2d& pos) -> double
	{
		return std::exp(-pos[0] - pos[1]);
	};

	int baseGrid = 32;
	int maxBaseGrid = baseGrid * int(std::pow(2,4));
	
	for (; baseGrid < maxBaseGrid; baseGrid *= 2)
	{
		double dx = PI / double(baseGrid);
		Vec2d origin = Vec2d::Zero();
		Vec2i size = Vec2i(std::round(PI / dx), std::round(PI / dx));
		Transform xform(dx, origin);

		AnalyticalPoissonSolver solver(xform, size);
		double error = solver.solve(rhs, solution);

		std::cout << "L-infinity error at " << baseGrid << "^2: " << error << std::endl;
	}
}
