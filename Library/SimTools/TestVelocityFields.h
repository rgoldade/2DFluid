#ifndef LIBRARY_TESTVELOCITYFIELDS_H
#define LIBRARY_TESTVELOCITYFIELDS_H

#include "Common.h"
#include "Noise.h"
#include "Renderer.h"

///////////////////////////////////
//
// TestVelocityFields.h
// Ryan Goldade 2018
//
////////////////////////////////////

////////////////////////////////////
//
// 2-D single vortex velocity field simulation
// from the standard surface tracker tests
//
////////////////////////////////////

class SingleVortexSim2D
{
public:
	SingleVortexSim2D() : myTime(0), myTotalTime(6)
	{}

	SingleVortexSim2D(Real t0, Real T) : myTime(t0), myTotalTime(T)
	{}

	void drawSimVectors(Renderer& renderer, Real dt = 1., Real radius = 10., const Vec3f& colour = Vec3f(0))
	{
		std::vector<Vec2R> startPoints;
		std::vector<Vec2R> endPoints;

		for (Real r = 0.; r <= radius; r += radius / 10.)
		{
			for (Real theta = 0.; theta < 2 * Util::PI; theta += Util::PI / 16.)
			{
				Vec2R p0 = Vec2R(r * cos(theta), r * sin(theta));
				Vec2R p1 = p0 + dt * (*this)(dt, p0);

				startPoints.push_back(p0);
				endPoints.push_back(p1);
			}
		}
		renderer.addLines(startPoints, endPoints, colour);
	}

	void advance(double dt)
	{
		myTime += dt;
	}

	// Sample positions given in world space
	Vec2R operator()(Real dt, const Vec2R& pos) const
	{
		double sinX = sin(Util::PI * pos[0]);
		double sinY = sin(Util::PI * pos[1]);

		double cosX = cos(Util::PI * pos[0]);
		double cosY = cos(Util::PI * pos[1]);

		double u = 2.0 * sinY * cosY * sinX * sinX;
		double v = -2.0 * sinX * cosX * sinY * sinY;

		return Vec2R(u, v) * cos(Util::PI * (myTime + dt) / myTotalTime);
	}

private:

	Real myTime, myTotalTime;
};

///////////////////////////////////
//
// Curl noise velocity field generator.
// Adapted from El Topo
//
////////////////////////////////////

class CurlNoise2D
{
public:
	CurlNoise2D()
		: myNoiseLengthScale(1),
		myNoiseGain(1),
		myNoiseFunction(),
		myDx(1E-4)
	{
		myNoiseLengthScale[0] = 1.5;
		myNoiseGain[0] = 1.3;
	}

	Vec2R getVelocity(const Vec2R &x) const
	{
		Vec2R vel;
		vel[0] = ((potential(x[0], x[1] + myDx)[2] - potential(x[0], x[1] - myDx)[2])
			- (potential(x[0], x[1], myDx)[1] - potential(x[0], x[1], -myDx)[1])) / (2 * myDx);
		vel[1] = ((potential(x[0], x[1], myDx)[0] - potential(x[0], x[1], -myDx)[0])
			- (potential(x[0] + myDx, x[1])[2] - potential(x[0] - myDx, x[1])[2])) / (2 * myDx);

		return vel;
	}

	Vec3R potential(Real x, Real y, Real z = 0.) const
	{
		Vec3R psi(0, 0, 0);
		Real heightFactor = 0.5;

		Vec3R centre(0.0, 1.0, 0.0);
		Real radius = 4.0;

		for (unsigned int i = 0; i<myNoiseLengthScale.size(); ++i)
		{
			Real sx = x / myNoiseLengthScale[i];
			Real sy = y / myNoiseLengthScale[i];
			Real sz = z / myNoiseLengthScale[i];

			Vec3R psi_i(0.f, 0.f, noise2(sx, sy, sz));

			Real dist = mag(Vec3R(x, y, z) - centre);
			Real scale = Util::max((radius - dist) / radius, Real(0));
			psi_i *= scale;

			psi += heightFactor * myNoiseGain[i] * psi_i;
		}

		return psi;
	}

	Real noise2(Real x, Real y, Real z) const { return myNoiseFunction(z - 203.994, x + 169.47, y - 205.31); }

	inline Vec2R operator()(Real, const Vec2R& pos) const
	{
		return getVelocity(pos);
	}

private:

	std::vector<Real> myNoiseLengthScale, myNoiseGain;
	Real myDx;

	FlowNoise3 myNoiseFunction;
};

///////////////////////////////////
//
// Child class fluid sim that just creates a velocity field that
// rotates CCW. Useful sanity check for surface trackers.
//
////////////////////////////////////

class CircularSim2D
{
public:
	CircularSim2D() : myCenter(Vec2R(0)), myScale(1.) {}
	CircularSim2D(const Vec2R& c, Real scale = 1.) : myCenter(c), myScale(scale) {}

	void drawSimVectors(Renderer& renderer, Real dt = 1., Real radius = 10., const Vec3f& colour = Vec3f(0))
	{
		std::vector<Vec2R> startPoints;
		std::vector<Vec2R> endPoints;

		for (Real r = 0.0; r <= radius; r += radius / 10.)
		{
			for (Real theta = 0.0; theta < 2 * Util::PI; theta += Util::PI / 16.0)
			{
				Vec2R startPoint = Vec2R(r * cos(theta), r * sin(theta)) + myCenter;
				Vec2R endPoint = startPoint + dt * (*this)(dt, startPoint);

				startPoints.push_back(startPoint);
				endPoints.push_back(endPoint);
			}
		}
		renderer.addLines(startPoints, endPoints, colour);
	};

	inline Vec2R operator()(Real, const Vec2R& pos) const
	{
		return Vec2R(pos[1] - myCenter[1], -(pos[0] - myCenter[0])) * myScale;
	}

private:
	Vec2R myCenter;
	Real myScale;
};

///////////////////////////////////
//
//
// Zalesak disk velocity field simulation
// Assumes you're running the notched disk IC
//
////////////////////////////////////

class NotchedDiskSim2D
{
public:

	void drawSimVectors(Renderer& renderer, Real dt = 1., Real radius = 10., const Vec3f& colour = Vec3f(0))
	{
		std::vector<Vec2R> startPoints;
		std::vector<Vec2R> endPoints;

		for (Real r = 0.0; r <= radius; r += radius / 10.)
		{
			for (Real theta = 0.0; theta < 2 * Util::PI; theta += Util::PI / 16.0)
			{
				Vec2R startPoint = Vec2R(r * cos(theta), r * sin(theta));
				Vec2R endPoint = startPoint + dt * (*this)(dt, startPoint);

				startPoints.push_back(startPoint);
				endPoints.push_back(endPoint);
			}
		}
		renderer.addLines(startPoints, endPoints, colour);
	};

	// Procedural velocity field
	inline Vec2R operator()(Real, const Vec2R& pos) const
	{
		return Vec2R((Util::PI / 314.) * (50.0 - pos[1]), (Util::PI / 314.) * (pos[0] - 50.0));
	}
};

#endif