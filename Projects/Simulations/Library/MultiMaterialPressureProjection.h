#ifndef SIMULATIONS_MULTIMATERIALPRESSUREPROJECTION_H
#define SIMULATIONS_MULTIMATERIALPRESSUREPROJECTION_H

#include "Common.h"
#include "ComputeWeights.h"
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

namespace PressureCellLabels
{
	constexpr int UNSOLVED_CELL = -1;
	constexpr int FLUID_CELL = 0;
}

class MultiMaterialPressureProjection
{
public:
    MultiMaterialPressureProjection(const std::vector<LevelSet> &surface,
									const std::vector<Real> &density,
									const LevelSet &solidSurface)
    : mySurfaceList(surface)

    , myDensityList(density)
    , mySolidSurface(solidSurface)
    , myMaterialsCount(surface.size())
    {
		assert(mySurfaceList.size() == myDensityList.size());

		for (int material = 1; material < myMaterialsCount; ++material)
			assert(mySurfaceList[material - 1].isGridMatched(mySurfaceList[material]));

		assert(mySurfaceList[0].isGridMatched(mySolidSurface));

		myPressure = ScalarGrid<Real>(mySolidSurface.xform(), mySolidSurface.size(), 0);
		myValidFaces = VectorGrid<MarkedCells>(mySolidSurface.xform(), mySolidSurface.size(), MarkedCells::UNVISITED, VectorGridSettings::SampleType::STAGGERED);

		myValidMaterialFacesList.resize(myMaterialsCount);
		for (int material = 0; material < myMaterialsCount; ++material)
			myValidMaterialFacesList[material] = VectorGrid<MarkedCells>(mySolidSurface.xform(), mySolidSurface.size(), MarkedCells::UNVISITED, VectorGridSettings::SampleType::STAGGERED);
		
		// Compute cut-cell weights
		mySolidCutCellWeights = computeCutCellWeights(solidSurface);
		
		myMaterialCutCellWeightsList.resize(myMaterialsCount);

		for (int material = 0; material < myMaterialsCount; ++material)
			myMaterialCutCellWeightsList[material] = computeCutCellWeights(surface[material]);

		// Now normalize the weights, removing the solid boundary contribution first.
		for (int axis : {0, 1})
		{
			forEachVoxelRange(Vec2i(0), myValidFaces.size(axis), [&](const Vec2i& face)
			{
				Real weight = 1;
				weight -= mySolidCutCellWeights(face, axis);
				weight = Util::clamp(weight, 0., weight);

				if (weight > 0)
				{
					Real accumulatedWeight = 0;
					for (int material = 0; material < myMaterialsCount; ++material)
						accumulatedWeight += myMaterialCutCellWeightsList[material](face, axis);

					if (accumulatedWeight > 0)
					{
						weight /= accumulatedWeight;

						for (int material = 0; material < myMaterialsCount; ++material)
							myMaterialCutCellWeightsList[material](face, axis) *= weight;
					}
				}
				else
				{
					for (int material = 0; material < myMaterialsCount; ++material)
						myMaterialCutCellWeightsList[material](face, axis) = 0;
				}

				// Debug check
				Real totalWeight = mySolidCutCellWeights(face, axis);

				for (int material = 0; material < myMaterialsCount; ++material)
					totalWeight += myMaterialCutCellWeightsList[material](face, axis);

				if (totalWeight == 0)
				{
					// If there is a zero total weight it is likely due to a fluid-fluid boundary
					// falling exactly across a grid face. There should never be a zero weight
					// along a fluid-solid boundary.

					std::vector<int> faceAlignedSurfaces;

					int otherAxis = (axis + 1) % 2;

					Vec2R offset(0); offset[otherAxis] = .5;

					for (int material = 0; material < myMaterialsCount; ++material)
					{
						Vec2R pos0 = myMaterialCutCellWeightsList[material].indexToWorld(Vec2R(face) - offset, axis);
						Vec2R pos1 = myMaterialCutCellWeightsList[material].indexToWorld(Vec2R(face) + offset, axis);

						Real weight = lengthFraction(surface[material].interp(pos0), surface[material].interp(pos1));

						if (weight == 0)
							faceAlignedSurfaces.push_back(material);
					}

					if (!(faceAlignedSurfaces.size() > 1))
					{
						std::cout << "Zero weight problems!!" << std::endl;
						exit(-1);
					}
					assert(faceAlignedSurfaces.size() > 1);

					myMaterialCutCellWeightsList[faceAlignedSurfaces[0]](face, axis) = 1.;
				}
			});
		}
    }

	const VectorGrid<MarkedCells>& getValidFaces(int material)
	{
		assert(material < myMaterialsCount);
		return myValidMaterialFacesList[material];
	}

    void project(std::vector<VectorGrid<Real>> &velocities);

	void drawPressure(Renderer &renderer) const;

	void printPressure(const std::string &filename) const
	{
		myPressure.printAsOBJ(filename + ".obj");
	}

private:

    ScalarGrid<Real> myPressure;
    VectorGrid<MarkedCells> myValidFaces;

    UniformGrid<int> myMaterialLabels;
    UniformGrid<int> mySolverIndex;

    const std::vector<LevelSet> &mySurfaceList;
    const std::vector<Real> &myDensityList;

	VectorGrid<Real> mySolidCutCellWeights;
	std::vector<VectorGrid<Real>> myMaterialCutCellWeightsList;
	std::vector<VectorGrid<MarkedCells>> myValidMaterialFacesList;

    const LevelSet &mySolidSurface;
    const int myMaterialsCount;
};

#endif