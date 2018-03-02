#pragma once
#ifdef _MSC_VER
	#include <Windows.h>
#else
	#include <sys/time.h>
#endif 

#include "util.h"

///////////////////////////////////
//
// Timer.h
// Ryan Goldade 2017
//
// 
////////////////////////////////////

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
		gettimeofday(&m_start, &tz);
#endif
	}

	inline Real stop()
	{
#ifdef _MSC_VER
		QueryPerformanceCounter(&EndingTime);
		ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

		return (Real)ElapsedMicroseconds.QuadPart / (Real) Frequency.QuadPart;
#else

		struct timezone tz;
		gettimeofday(&m_end, &tz);

		return (m_end.tv_sec + m_end.tv_usec * 1E-6) - (m_start.tv_sec + m_start.tv_usec * 1E-6);
#endif
	}

	inline void reset()
	{
#ifdef _MSC_VER
		QueryPerformanceFrequency(&Frequency);
		QueryPerformanceCounter(&StartingTime);
#else
		struct timezone tz;
		gettimeofday(&m_start, &tz);
#endif
	}

private:

#ifdef _MSC_VER
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
#else
	struct timeval m_start, m_end;
#endif
};

