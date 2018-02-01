#pragma once

///////////////////////////////////
//
// Integrator.h
// Ryan Goldade 2016
//
// Integration functions to assist 
// with Lagrangian advection.
//
////////////////////////////////////

namespace Integrator
{
	// FuncField is assumed to be up-to-date, i.e. internal time = t(n)
	// where t(n+1) = t(n) + h
	template<typename T, typename FuncField>
	struct forward_euler
	{
		inline T operator()(const T& x, Real h, const FuncField& f) const
		{
			return x + h * f(0, x);
		}
	};

	template<typename T, typename FuncField>
	struct rk3
	{
		inline T operator()(const T& x, Real h, const FuncField& f) const
		{
			T k1 = h * f(0., x);
			T k2 = h * f(h / 2., x + k1 / 2.);
			T k3 = h * f(h, x - k1 + k2);

			return x + 1. / 6. * (k1 + 4.*k2 + k3);
		}
	};
}