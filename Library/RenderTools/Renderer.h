#ifndef FLUIDSIM2D_RENDERER_H
#define FLUIDSIM2D_RENDERER_H

#include <functional>
#include <vector>

#ifdef __APPLE__
	#include <GLUT/glut.h>
#else
	#include <GL/glut.h>
#endif

#include "Utilities.h"

///////////////////////////////////
//
// Renderer.h
// Ryan Goldade 2016
//
// Render machine to handle adding
// primitives from various different
// sources and having a central place
// to run the render loop.
//
////////////////////////////////////

namespace FluidSim2D
{

class Renderer
{
public:
	Renderer(const char *title, Vec2i windowSize, Vec2d screenOrigin,
				double screenHeight, int *argc, char **argv);

	void display();
	void mouse(int button, int state, int x, int y);
	void drag(int x, int y);
	void keyboard(unsigned char key, int x, int y);
	void reshape(int w, int h);

	void setUserMouseClick(const std::function<void(int, int, int, int)>& mouseClickFunction);
	void setUserKeyboard(const std::function<void(unsigned char, int, int)>& keyboardFunction);
	void setUserMouseDrag(const std::function<void(int, int)>& mouseDragFunction);
	void setUserDisplay(const std::function<void()>& displayFunction);

	void addPoint(const Vec2d& point, const Vec3d& colour = Vec3d(0, 0, 0), double size = 1.);
	void addPoints(const VecVec2d& points, const Vec3d& colour = Vec3d(0), double size = 1.);

	void addLine(const Vec2d& startingPoint, const Vec2d& endingPoint, const Vec3d& colour = Vec3d(0), double lineWidth = 1);
	void addLines(const VecVec2d& startingPoints, const VecVec2d& endingPoints, const Vec3d& colour, double lineWidth = 1);

	void addTriFaces(const VecVec2d& vertices, const VecVec3i& faces, const VecVec3d& faceColours);
	void addQuadFaces(const VecVec2d& vertices, const VecVec4i& faces, const VecVec3d& faceColours);

	void drawPrimitives() const;

	void printImage(const std::string& filename) const;

	void clear();
	void run();

private:

	std::vector<VecVec2d> myPoints;
	VecVec3d myPointColours;
	std::vector<double> myPointSizes;

	std::vector<VecVec2d> myLineStartingPoints;
	std::vector<VecVec2d> myLineEndingPoints;
	VecVec3d myLineColours;
	std::vector<double> myLineWidths;

	std::vector<VecVec2d> myTriVertices;
	std::vector<VecVec3i> myTriFaces;
	std::vector<VecVec3d> myTriFaceColours;

	std::vector<VecVec2d> myQuadVertices;
	std::vector<VecVec4i> myQuadFaces;
	std::vector<VecVec3d> myQuadFaceColours;

	// width, height
	Vec2i myWindowSize;

	Vec2d myCurrentScreenOrigin;
	double myCurrentScreenHeight;

	Vec2d myDefaultScreenOrigin;
	double myDefaultScreenHeight;

	// Mouse specific state
	Vec2i myMousePosition;
	bool myMouseMoved;

	enum class MouseAction { INACTIVE, PAN, ZOOM_IN, ZOOM_OUT };
	MouseAction myMouseAction;

	// User specific extensions for each glut callback
	std::function<void(unsigned char, int, int)> myUserKeyboardFunction;
	std::function<void(int, int, int, int)> myUserMouseClickFunction;
	std::function<void(int, int)> myUserMouseDragFunction;
	std::function<void()> myUserDisplayFunction;
};

}

#endif