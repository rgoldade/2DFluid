#ifndef LIBRARY_TEST_VELOCITY_FIELD_H
#define LIBRARY_TEST_VELOCITY_FIELD_H

#include "Noise.h"
#include "Renderer.h"
#include "Utilities.h"

///////////////////////////////////
//
// TestVelocityFields.h
// Ryan Goldade 2018
//
////////////////////////////////////

namespace FluidSim2D::SimTools
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

	SingleVortexField(float startTime, float deformationPeriod)
		: mySimTime(startTime)
		, myDeformationPeriod(deformationPeriod)
	{}

	void drawSimVectors(Renderer& renderer, float dt = 1., float radius = 10., const Vec3f& colour = Vec3f(0))
	{
		std::vector<Vec2f> startPoints;
		std::vector<Vec2f> endPoints;

		for (float drawRadius = 0.; drawRadius <= radius; drawRadius += radius / 10.)
			for (float theta = 0.; theta < 2 * PI; theta += PI / 16.)
			{
				Vec2f start(drawRadius * std::cos(theta), drawRadius * std::sin(theta));
				Vec2f end = start + dt * (*this)(dt, start);

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
	Vec2f operator()(float dt, const Vec2f& pos) const
	{
		float sinX = sin(PI * pos[0]);
		float sinY = sin(PI * pos[1]);

		float cosX = cos(PI * pos[0]);
		float cosY = cos(PI * pos[1]);

		double u = 2.0 * sinY * cosY * sinX * sinX;
		double v = -2.0 * sinX * cosX * sinY * sinY;

		return Vec2f(u, v) * cos(PI * (mySimTime + dt) / myDeformationPeriod);
	}

private:
	float mySimTime, myDeformationPeriod;
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

	Vec2f getVelocity(const Vec2f& x) const
	{
		Vec2f vel;
		vel[0] = ((potential(x[0], x[1] + myDx)[2] - potential(x[0], x[1] - myDx)[2])
			- (potential(x[0], x[1], myDx)[1] - potential(x[0], x[1], -myDx)[1])) / (2 * myDx);
		vel[1] = ((potential(x[0], x[1], myDx)[0] - potential(x[0], x[1], -myDx)[0])
			- (potential(x[0] + myDx, x[1])[2] - potential(x[0] - myDx, x[1])[2])) / (2 * myDx);

		return vel;
	}

	Vec2f operator()(float, const Vec2f& pos) const
	{
		return getVelocity(pos);
	}

private:

	Vec3f potential(float x, float y, float z = 0.) const
	{
		Vec3f psi(0, 0, 0);
		float heightFactor = 0.5;

		Vec3f centre(0.0, 1.0, 0.0);
		float radius = 4.0;

		for (int i = 0; i < myNoiseLengthScale.size(); ++i)
		{
			float sx = x / myNoiseLengthScale[i];
			float sy = y / myNoiseLengthScale[i];
			float sz = z / myNoiseLengthScale[i];

			Vec3f psi_i(0, 0, noise2(sx, sy, sz));

			float dist = mag(Vec3f(x, y, z) - centre);
			float scale = std::max((radius - dist) / radius, float(0));
			psi_i *= scale;

			psi += heightFactor * myNoiseGain[i] * psi_i;
		}

		return psi;
	}

	float noise2(float x, float y, float z) const { return myNoiseFunction(z - 203.994, x + 169.47, y - 205.31); }

	std::vector<float> myNoiseLengthScale, myNoiseGain;
	float myDx;

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
	CircularField() : myCenter(Vec2f(0)), myScale(1.) {}
	CircularField(const Vec2f& center, float scale = 1.) : myCenter(center), myScale(scale) {}

	void drawSimVectors(Renderer& renderer, float dt = 1., float radius = 10., const Vec3f& colour = Vec3f(0))
	{
		std::vector<Vec2f> startPoints;
		std::vector<Vec2f> endPoints;

		for (float drawRadius = 0.0; drawRadius <= radius; drawRadius += radius / 10.)
			for (float theta = 0.0; theta < 2 * PI; theta += PI / 16.0)
			{
				Vec2f startPoint = Vec2f(drawRadius * cos(theta), drawRadius * sin(theta)) + myCenter;
				Vec2f endPoint = startPoint + dt * (*this)(dt, startPoint);

				startPoints.push_back(startPoint);
				endPoints.push_back(endPoint);
			}

		renderer.addLines(startPoints, endPoints, colour);
	};

	Vec2f operator()(float, const Vec2f& pos) const
	{
		return Vec2f(pos[1] - myCenter[1], -(pos[0] - myCenter[0])) * myScale;
	}

private:
	Vec2f myCenter;
	float myScale;
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

	void drawSimVectors(Renderer& renderer, float dt = 1., float radius = 10., const Vec3f& colour = Vec3f(0))
	{
		std::vector<Vec2f> startPoints;
		std::vector<Vec2f> endPoints;

		for (float drawRadius = 0; drawRadius <= radius; drawRadius += radius / 10.)
			for (float theta = 0; theta < 2 * PI; theta += PI / 16.)
			{
				Vec2f startPoint = Vec2f(drawRadius * cos(theta), drawRadius * sin(theta));
				Vec2f endPoint = startPoint + dt * (*this)(dt, startPoint);

				startPoints.push_back(startPoint);
				endPoints.push_back(endPoint);
			}

		renderer.addLines(startPoints, endPoints, colour);
	};

	// Procedural velocity field
	Vec2f operator()(float, const Vec2f& pos) const
	{
		return Vec2f((PI / 314.) * (50.0 - pos[1]), (PI / 314.) * (pos[0] - 50.0));
	}
};

}

#endif