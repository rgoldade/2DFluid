#ifndef FLUIDSIM2D_TIMER_H
#define FLUIDSIM2D_TIMER_H

#ifdef _MSC_VER
	#include <Windows.h>
#else
	#include <sys/time.h>
#endif 

#include "Utilities.h"

///////////////////////////////////
//
// Timer.h
// Ryan Goldade 2017
//
// 
////////////////////////////////////

namespace FluidSim2D
{

class Timer
{
public:
	Timer()
	{
#ifdef _MSC_VER
		QueryPerformanceFrequency(&Frequency);
		QueryPerformanceCounter(&StartingTime);
#else
		struct timezone tz;
		gettimeofday(&myStart, &tz);
#endif
	}

	float stop()
	{
#ifdef _MSC_VER
		QueryPerformanceCounter(&EndingTime);
		ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

		return float(ElapsedMicroseconds.QuadPart) / float(Frequency.QuadPart);
#else

		struct timezone tz;
		gettimeofday(&myEnd, &tz);

		return (myEnd.tv_sec + myEnd.tv_usec * 1E-6) - (myStart.tv_sec + myStart.tv_usec * 1E-6);
#endif
	}

	void reset()
	{
#ifdef _MSC_VER
		QueryPerformanceFrequency(&Frequency);
		QueryPerformanceCounter(&StartingTime);
#else
		struct timezone tz;
		gettimeofday(&myStart, &tz);
#endif
	}

private:

#ifdef _MSC_VER
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
#else
	struct timeval myStart, myEnd;
#endif
};

}
#endif