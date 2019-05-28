#ifndef SIMULATIONS_MULTIMATERIALPRESSUREPROJECTION_H
#define SIMULATIONS_MULTIMATERIALPRESSUREPROJECTION_H

#include "Common.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "UniformGrid.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// MultiMaterialPressureProjection.h/cpp
// Ryan Goldade 2017
//
////////////////////////////////////

static constexpr int UNSOLVED = -1;

class MultiMaterialPressureProjection
{
public:
    MultiMaterialPressureProjection(const std::vector<LevelSet> &surface,
				    const std::vector<VectorGrid<Real>> &velocity,
				    const std::vector<Real> &density,
				    const LevelSet &collision)
    : mySurfaceList(surface)
    , myVelocityList(velocity)
    , myDensityList(density)
    , mySolidSurface(collision)
    , myMaterialsCount(surface.size())
    {
		assert(mySurfaceList.size() == myVelocityList.size() &&
				mySurfaceList.size() == myDensityList.size());

		for (int material = 1; material < myMaterialsCount; ++material)
		{
			assert(mySurfaceList[material - 1].isGridMatched(mySurfaceList[material]));
			assert(myVelocityList[material - 1].isGridMatched(myVelocityList[material]));
		}

		// Since every surface and every velocity field is matched, we only need to compare
		// on pair of fields to make sure the two sets are matched.

		assert(myVelocityList[0].size(0)[0] - 1 == mySurfaceList[0].size()[0] &&
				myVelocityList[0].size(0)[1] == mySurfaceList[0].size()[1] &&
				myVelocityList[0].size(1)[0] == mySurfaceList[0].size()[0] &&
				myVelocityList[0].size(1)[1] - 1 == mySurfaceList[0].size()[1]);

		assert(mySurfaceList[0].isGridMatched(mySolidSurface));

		myPressure = ScalarGrid<Real>(mySolidSurface.xform(), mySolidSurface.size(), 0);
		myValid = VectorGrid<Real>(mySolidSurface.xform(), mySolidSurface.size(), 0, VectorGridSettings::SampleType::STAGGERED);
    }

    void project(const std::vector<VectorGrid<Real>> &materialCutCellWeights,
					const VectorGrid<Real> &collisionCutCellWeights);

    void applySolution(std::vector<VectorGrid<Real>> &velocity,
						const std::vector<VectorGrid<Real>> &materialCutCellWeights) const;

	void drawPressure(Renderer &renderer) const;

private:

    ScalarGrid<Real> myPressure;
    VectorGrid<Real> myValid;

    UniformGrid<int> myMaterialLabels;
    UniformGrid<int> mySolverIndex;

    const std::vector<LevelSet> &mySurfaceList;
    const std::vector<VectorGrid<Real>> &myVelocityList;
    const std::vector<Real> &myDensityList;

    const LevelSet &mySolidSurface;
    const int myMaterialsCount;
};

#endif