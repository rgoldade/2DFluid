#ifndef LIBRARY_RENDERER_H
#define LIBRARY_RENDERER_H

#include <functional>
#include <vector>

#ifdef __APPLE__
	#include <GLUT/glut.h>
#else
	#include <GL/glut.h>
#endif

#include "Utilities.h"
#include "Vec.h"

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

namespace FluidSim2D::RenderTools
{

using namespace Utilities;

class Renderer
{
public:
	Renderer(const char *title, Vec2i windowSize, Vec2f screenOrigin,
				float screenHeight, int *argc, char **argv);

	void display();
	void mouse(int button, int state, int x, int y);
	void drag(int x, int y);
	void keyboard(unsigned char key, int x, int y);
	void reshape(int w, int h);

	void setUserMouseClick(const std::function<void(int, int, int, int)>& mouseClickFunction);
	void setUserKeyboard(const std::function<void(unsigned char, int, int)>& keyboardFunction);
	void setUserMouseDrag(const std::function<void(int, int)>& mouseDragFunction);
	void setUserDisplay(const std::function<void()>& displayFunction);

	void addPoint(const Vec2f& point, const Vec3f& colour = Vec3f(0, 0, 0), float size = 1.);
	void addPoints(const std::vector<Vec2f>& points, const Vec3f& colour = Vec3f(0), float size = 1.);

	void addLine(const Vec2f& startingPoint, const Vec2f& endingPoint, const Vec3f& colour = Vec3f(0), float lineWidth = 1);
	void addLines(const std::vector<Vec2f>& startingPoints, const std::vector<Vec2f>& endingPoints, const Vec3f& colour, float lineWidth = 1);

	void addTriFaces(const std::vector<Vec2f>& vertices, const std::vector<Vec3i>& faces, const std::vector<Vec3f>& faceColours);
	void addQuadFaces(const std::vector<Vec2f>& vertices, const std::vector<Vec4i>& faces, const std::vector<Vec3f>& faceColours);

	void drawPrimitives() const;

	void printImage(const std::string& filename) const;

	void clear();
	void run();

private:

	std::vector<std::vector<Vec2f>> myPoints;
	std::vector<Vec3f> myPointColours;
	std::vector<float> myPointSizes;

	std::vector<std::vector<Vec2f>> myLineStartingPoints;
	std::vector<std::vector<Vec2f>> myLineEndingPoints;
	std::vector<Vec3f> myLineColours;
	std::vector<float> myLineWidths;

	std::vector<std::vector<Vec2f>> myTriVertices;
	std::vector<std::vector<Vec3i>> myTriFaces;
	std::vector<std::vector<Vec3f>> myTriFaceColours;

	std::vector<std::vector<Vec2f>> myQuadVertices;
	std::vector<std::vector<Vec4i>> myQuadFaces;
	std::vector<std::vector<Vec3f>> myQuadFaceColours;

	// width, height
	Vec2i myWindowSize;

	Vec2f myCurrentScreenOrigin;
	float myCurrentScreenHeight;

	Vec2f myDefaultScreenOrigin;
	float myDefaultScreenHeight;

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