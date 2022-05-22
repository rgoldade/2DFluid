#ifndef FLUIDSIM2D_TEST_VELOCITY_FIELD_H
#define FLUIDSIM2D_TEST_VELOCITY_FIELD_H

#include "Noise.h"
#include "Renderer.h"
#include "Utilities.h"

///////////////////////////////////
//
// TestVelocityFields.h
// Ryan Goldade 2018
//
////////////////////////////////////

namespace FluidSim2D
{

////////////////////////////////////
//
// 2-D single vortex velocity field simulation
// from the standard surface tracker tests
//
////////////////////////////////////

class SingleVortexField
{
public:
	SingleVortexField() : mySimTime(0), myDeformationPeriod(6)
	{}

	SingleVortexField(double startTime, double deformationPeriod)
		: mySimTime(startTime)
		, myDeformationPeriod(deformationPeriod)
	{}

	void drawSimVectors(Renderer& renderer, double dt = 1, double radius = 10, const Vec3d& colour = Vec3d::Zero())
	{
		VecVec2d startPoints;
		VecVec2d endPoints;

		for (double drawRadius = 0.; drawRadius <= radius; drawRadius += radius / 10.)
			for (double theta = 0.; theta < 2. * PI; theta += PI / 16.)
			{
				Vec2d start(drawRadius * std::cos(theta), drawRadius * std::sin(theta));
				Vec2d end = start + dt * (*this)(dt, start);

				startPoints.push_back(start);
				endPoints.push_back(end);
			}

		renderer.addLines(startPoints, endPoints, colour);
	}

	void advance(double dt)
	{
		mySimTime += dt;
	}

	// Sample positions given in world space
	Vec2d operator()(double dt, const Vec2d& pos) const
	{
		double sinX = sin(PI * pos[0]);
		double sinY = sin(PI * pos[1]);

		double cosX = cos(PI * pos[0]);
		double cosY = cos(PI * pos[1]);

		double u = 2. * sinY * cosY * sinX * sinX;
		double v = -2. * sinX * cosX * sinY * sinY;

		return Vec2d(u, v) * cos(PI * (mySimTime + dt) / myDeformationPeriod);
	}

private:
	double mySimTime, myDeformationPeriod;
};

///////////////////////////////////
//
// Curl noise velocity field generator.
// Adapted from El Topo
//
////////////////////////////////////

class CurlNoiseField
{
public:
	CurlNoiseField()
		: myNoiseLengthScale(1),
		myNoiseGain(1),
		myNoiseFunction(),
		myDx(1E-4)
	{
		myNoiseLengthScale[0] = 1.5;
		myNoiseGain[0] = 1.3;
	}

	Vec2d getVelocity(const Vec2d& x) const
	{
		Vec2d vel;
		vel[0] = ((potential(x[0], x[1] + myDx)[2] - potential(x[0], x[1] - myDx)[2])
			- (potential(x[0], x[1], myDx)[1] - potential(x[0], x[1], -myDx)[1])) / (2 * myDx);
		vel[1] = ((potential(x[0], x[1], myDx)[0] - potential(x[0], x[1], -myDx)[0])
			- (potential(x[0] + myDx, x[1])[2] - potential(x[0] - myDx, x[1])[2])) / (2 * myDx);

		return vel;
	}

	Vec2d operator()(double, const Vec2d& pos) const
	{
		return getVelocity(pos);
	}

private:

	Vec3d potential(double x, double y, double z = 0.) const
	{
		Vec3d psi(0, 0, 0);
		double heightFactor = 0.5;

		Vec3d centre(0.0, 1.0, 0.0);
		double radius = 4.0;

		for (int i = 0; i < myNoiseLengthScale.size(); ++i)
		{
			double sx = x / myNoiseLengthScale[i];
			double sy = y / myNoiseLengthScale[i];
			double sz = z / myNoiseLengthScale[i];

			Vec3d psi_i(0, 0, noise2(sx, sy, sz));

			double dist = (Vec3d(x, y, z) - centre).norm();
			double scale = std::max((radius - dist) / radius, double(0));
			psi_i *= scale;

			psi += heightFactor * myNoiseGain[i] * psi_i;
		}

		return psi;
	}

	double noise2(double x, double y, double z) const { return myNoiseFunction(z - 203.994, x + 169.47, y - 205.31); }

	std::vector<double> myNoiseLengthScale, myNoiseGain;
	double myDx;

	FlowNoise3 myNoiseFunction;
};

///////////////////////////////////
//
// Child class fluid sim that just creates a velocity field that
// rotates CCW. Useful sanity check for surface trackers.
//
////////////////////////////////////

class CircularField
{
public:
	CircularField() : myCenter(Vec2d(0)), myScale(1.) {}
	CircularField(const Vec2d& center, double scale = 1.) : myCenter(center), myScale(scale) {}

	void drawSimVectors(Renderer& renderer, double dt = 1., double radius = 10., const Vec3d& colour = Vec3d(0))
	{
		VecVec2d startPoints;
		VecVec2d endPoints;

		for (double drawRadius = 0.0; drawRadius <= radius; drawRadius += radius / 10.)
			for (double theta = 0.0; theta < 2 * PI; theta += PI / 16.0)
			{
				Vec2d startPoint = Vec2d(drawRadius * cos(theta), drawRadius * sin(theta)) + myCenter;
				Vec2d endPoint = startPoint + dt * (*this)(dt, startPoint);

				startPoints.push_back(startPoint);
				endPoints.push_back(endPoint);
			}

		renderer.addLines(startPoints, endPoints, colour);
	};

	Vec2d operator()(double, const Vec2d& pos) const
	{
		return Vec2d(pos[1] - myCenter[1], -(pos[0] - myCenter[0])) * myScale;
	}

private:
	Vec2d myCenter;
	double myScale;
};

///////////////////////////////////
//
// Zalesak disk velocity field simulation
// Assumes you're running the notched disk IC
//
////////////////////////////////////

class NotchedDiskField
{
public:

	void drawSimVectors(Renderer& renderer, double dt = 1., double radius = 10., const Vec3d& colour = Vec3d(0))
	{
		VecVec2d startPoints;
		VecVec2d endPoints;

		for (double drawRadius = 0; drawRadius <= radius; drawRadius += radius / 10.)
			for (double theta = 0; theta < 2 * PI; theta += PI / 16.)
			{
				Vec2d startPoint = Vec2d(drawRadius * cos(theta), drawRadius * sin(theta));
				Vec2d endPoint = startPoint + dt * (*this)(dt, startPoint);

				startPoints.push_back(startPoint);
				endPoints.push_back(endPoint);
			}

		renderer.addLines(startPoints, endPoints, colour);
	};

	// Procedural velocity field
	Vec2d operator()(double, const Vec2d& pos) const
	{
		return Vec2d((PI / 314.) * (50.0 - pos[1]), (PI / 314.) * (pos[0] - 50.0));
	}
};

class DeformationField
{
public:

	// Procedural velocity field
	Vec2d operator()(double, const Vec2d& pos) const
	{
		return Vec2d(-std::sin(4 * PI * (pos[0] + .5)) * std::sin(4 * PI * (pos[1] + .5)), -std::cos(4 * PI * (pos[0] + .5)) * std::cos(4 * PI * (pos[1] + .5)));
	}
};



}

#endif