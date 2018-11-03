#pragma once

#include "Common.h"

///////////////////////////////////
//
// Integrator.h
// Ryan Goldade 2016
//
// Integration functions to assist 
// with Lagrangian advection.
//
////////////////////////////////////

enum class IntegrationOrder { FORWARDEULER, RK3 };

template<typename T, typename Function>
inline T Integrator(Real h, const T& x, const Function& f, IntegrationOrder order)
{
	T val;

	switch (order)
	{
	case IntegrationOrder::FORWARDEULER:
		val = x + h * f(0, x);
		break;
	case IntegrationOrder::RK3:
	{
		T k1 = h * f(0., x);
		T k2 = h * f(h / 2., x + k1 / 2.);
		T k3 = h * f(h, x - k1 + k2);

		val = x + (1. / 6.) * (k1 + 4. * k2 + k3);
		break;
	}
	default:
		assert(false);
	}

	return val;
}