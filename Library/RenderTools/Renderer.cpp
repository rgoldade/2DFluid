#include "Renderer.h"

#include "simple_svg_1.0.0.hpp"

namespace FluidSim2D
{

// Helper struct because glut is a pain.
// This is probably a very bad design choice
// but glut doesn't make life easy.
class GlutHelper
{
public:
	static void init(Renderer* render)
	{
		// Safety check to only ever take one instance
		// of a Renderer
		if (!myRender) myRender = render;

		glutReshapeFunc(reshape);
		glutDisplayFunc(display);
		glutMouseFunc(mouse);
		glutMotionFunc(drag);
		glutKeyboardFunc(keyboard);
	}
private:
	static void reshape(int w, int h)
	{
		if (myRender) myRender->reshape(w, h);
	}

	static void display()
	{
		if (myRender) myRender->display();
	}

	static void mouse(int button, int state, int x, int y)
	{
		if (myRender) myRender->mouse(button, state, x, y);
	}

	static void drag(int x, int y)
	{
		if (myRender) myRender->drag(x, y);
	}

	static void keyboard(unsigned char key, int x, int y)
	{
		if (myRender) myRender->keyboard(key, x, y);
	}

	static Renderer* myRender;
};

Renderer* GlutHelper::myRender;

Renderer::Renderer(const char *title, Vec2i windowSize, Vec2d screenOrigin,
	double screenHeight, int *argc, char **argv)
	: myWindowSize(windowSize)
	, myCurrentScreenOrigin(screenOrigin)
	, myCurrentScreenHeight(screenHeight)
	, myDefaultScreenOrigin(screenOrigin)
	, myDefaultScreenHeight(screenHeight)
	, myMouseAction(MouseAction::INACTIVE)
{
	assert(windowSize[0] >= 0 && windowSize[1] >= 0);
	glutInit(argc, argv);

	//TODO: review if these flags are needed
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(myWindowSize[0], myWindowSize[1]);
	glutCreateWindow(title);
	glClearColor(1.0, 1.0, 1.0, 0.0);

	GlutHelper::init(this);
}

void Renderer::display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, GLsizei(myWindowSize[0]), GLsizei(myWindowSize[1]));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(myCurrentScreenOrigin[0], myCurrentScreenOrigin[0] + (myCurrentScreenHeight * double(myWindowSize[0])) / double(myWindowSize[1]), myCurrentScreenOrigin[1], myCurrentScreenOrigin[1] + myCurrentScreenHeight, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Run user specific display funcion
	if (myUserDisplayFunction) myUserDisplayFunction();

	drawPrimitives();
	glutSwapBuffers();
	glutPostRedisplay();
}

void Renderer::mouse(int button, int state, int x, int y)
{
	// If the user provided a custom mouse handler, then use that
	if (myUserMouseClickFunction) myUserMouseClickFunction(button, state, x, y);
	else
	{
		if (state == GLUT_UP)
		{
			double scale;
			switch (myMouseAction)
			{
				// PAN is only for drag
				// TODO: add click specific movement
			case MouseAction::ZOOM_IN:

				// Zoom in by 2x
				scale = .5;

				myCurrentScreenOrigin[0] -= .5 * (scale - 1.) * double(myWindowSize[0]) * myCurrentScreenHeight / double(myWindowSize[1]);
				myCurrentScreenOrigin[1] -= .5 * (scale - 1.) * myCurrentScreenHeight;
				myCurrentScreenHeight *= scale;

				glutPostRedisplay();
				break;

			case MouseAction::ZOOM_OUT:

				scale = 2.;
				myCurrentScreenOrigin[0] -= .5 * (scale - 1.) * double(myWindowSize[0]) * myCurrentScreenHeight / double(myWindowSize[1]);
				myCurrentScreenOrigin[1] -= .5 * (scale - 1.) * myCurrentScreenHeight;
				myCurrentScreenHeight *= scale;

				glutPostRedisplay();
			}

			myMouseAction = MouseAction::INACTIVE;
		}
		else
		{
			switch (button)
			{
			case GLUT_LEFT_BUTTON:
				myMouseAction = MouseAction::PAN;
				break;
			case GLUT_MIDDLE_BUTTON:
				myMouseAction = MouseAction::ZOOM_IN;
				break;
			case GLUT_RIGHT_BUTTON:
				myMouseAction = MouseAction::ZOOM_OUT;
			}
		}

		myMouseMoved = false; // Not currently used
		myMousePosition = Vec2i(x, y);
	}
}

void Renderer::drag(int x, int y)
{
	if (myUserMouseDragFunction) myUserMouseDragFunction(x, y);
	else
	{
		if (x != myMousePosition[0] || y != myMousePosition[1])
		{
			myMouseMoved = true;
			if (myMouseAction == MouseAction::PAN)
			{
				double pixelRatio = myCurrentScreenHeight / double(myWindowSize[1]);
				myCurrentScreenOrigin[0] -= pixelRatio * (double(x) - myMousePosition[0]);
				myCurrentScreenOrigin[1] += pixelRatio * (double(y) - myMousePosition[1]);
				glutPostRedisplay();
			}

			myMousePosition[0] = x;
			myMousePosition[1] = y;
		}
	}
}

void Renderer::keyboard(unsigned char key, int x, int y)
{
	if (myUserKeyboardFunction) myUserKeyboardFunction(key, x, y);

	// r triggers return to defaults
	if (key == 'r')
	{
		myCurrentScreenOrigin = myDefaultScreenOrigin;
		myCurrentScreenHeight = myDefaultScreenHeight;
		glutPostRedisplay();
	}
}

void Renderer::reshape(int w, int h)
{
	myWindowSize[0] = w;
	myWindowSize[1] = h;
	glutPostRedisplay();
}

void Renderer::setUserMouseClick(const std::function<void(int, int, int, int)>& mouseClickFunction)
{
	myUserMouseClickFunction = mouseClickFunction;
}

void Renderer::setUserKeyboard(const std::function<void(unsigned char, int, int)>& keyboardFunction)
{
	myUserKeyboardFunction = keyboardFunction;
}

void Renderer::setUserMouseDrag(const std::function<void(int, int)>& mouseDragFunction)
{
	myUserMouseDragFunction = mouseDragFunction;
}

void Renderer::setUserDisplay(const std::function<void()>& displayFunction)
{
	myUserDisplayFunction = displayFunction;
}

// These helpers make it easy to render out basic primitives without having to write
// a custom loop outside of this class
void Renderer::addPoint(const Vec2d& point, const Vec3d& colour, double size)
{
	myPoints.emplace_back(1, point);
	myPointColours.push_back(colour);
	myPointSizes.push_back(size);
}

void Renderer::addPoints(const VecVec2d& points, const Vec3d& colour, double size)
{
	myPoints.push_back(points);
	myPointColours.push_back(colour);
	myPointSizes.push_back(size);
}

void Renderer::addLine(const Vec2d& startingPoint, const Vec2d& endingPoint, const Vec3d& colour, double lineWidth)
{
	myLineStartingPoints.emplace_back(1, startingPoint);
	myLineEndingPoints.emplace_back(1, endingPoint);

	myLineColours.push_back(colour);
	myLineWidths.push_back(lineWidth);
}

void Renderer::addLines(const VecVec2d& startingPoints, const VecVec2d& endingPoints, const Vec3d& colour, double lineWidth)
{
	assert(startingPoints.size() == endingPoints.size());
	myLineStartingPoints.push_back(startingPoints);
	myLineEndingPoints.push_back(endingPoints);

	myLineColours.push_back(colour);
	myLineWidths.push_back(lineWidth);
}

void Renderer::addTriFaces(const VecVec2d& vertices, const VecVec3i& faces, const VecVec3d& faceColours)
{
	myTriVertices.push_back(vertices);
	myTriFaces.push_back(faces);
	myTriFaceColours.push_back(faceColours);
}

void Renderer::addQuadFaces(const VecVec2d& vertices, const VecVec4i& faces, const VecVec3d& faceColours)
{
	myQuadVertices.push_back(vertices);
	myQuadFaces.push_back(faces);
	myQuadFaceColours.push_back(faceColours);
}

void Renderer::drawPrimitives() const
{
	// Render quads
	assert(myQuadVertices.size() == myQuadFaces.size() &&
			myQuadVertices.size() == myQuadVertices.size());

	for (int quadListIndex = 0; quadListIndex < myQuadFaces.size(); ++quadListIndex)
	{
		assert(myQuadFaces[quadListIndex].size() == myQuadFaceColours[quadListIndex].size());

		for (int quadIndex = 0; quadIndex < myQuadFaces[quadListIndex].size(); ++quadIndex)
		{
			const Vec3t<GLfloat> quadFaceColour = myQuadFaceColours[quadListIndex][quadIndex].cast<GLfloat>();
			glColor3f(quadFaceColour[0], quadFaceColour[1], quadFaceColour[2]);

			glBegin(GL_QUADS);

			for (int pointIndex : {0, 1, 2, 3})
			{
				int quadVertexIndex = myQuadFaces[quadListIndex][quadIndex][pointIndex];

				assert(quadVertexIndex >= 0 && quadVertexIndex < myQuadVertices[quadListIndex].size());

				const Vec2t<GLfloat> vertexPoint = myQuadVertices[quadListIndex][quadVertexIndex].cast<GLfloat>();
				glVertex2f(vertexPoint[0], vertexPoint[1]);
			}

			glEnd();
		}
	}

	// Render tris
	assert(myTriVertices.size() == myTriFaces.size() &&
			myTriVertices.size() == myTriFaceColours.size());

	for (int triListIndex = 0; triListIndex < myTriFaces.size(); ++triListIndex)
	{
		assert(myTriFaces[triListIndex].size() == myTriFaceColours[triListIndex].size());

		for (int triIndex = 0; triIndex < myTriFaces[triListIndex].size(); ++triIndex)
		{
			const Vec3t<GLfloat> triFaceColour = myTriFaceColours[triListIndex][triIndex].cast<GLfloat>();
			glColor3f(triFaceColour[0], triFaceColour[1], triFaceColour[2]);

			glBegin(GL_TRIANGLES);

			for (int pointIndex : {0, 1, 2})
			{
				int triVertexIndex = myTriFaces[triListIndex][triIndex][pointIndex];

				assert(triVertexIndex >= 0 && triVertexIndex < myTriVertices[triListIndex].size());

				const Vec2t<GLfloat> vertexPoint = myTriVertices[triListIndex][triVertexIndex].cast<GLfloat>();
				glVertex2f(vertexPoint[0], vertexPoint[1]);
			}

			glEnd();
		}
	}

	assert(myLineStartingPoints.size() == myLineEndingPoints.size() &&
			myLineStartingPoints.size() == myLineColours.size() &&
			myLineStartingPoints.size() == myLineWidths.size());

	for (int lineListIndex = 0; lineListIndex < myLineStartingPoints.size(); ++lineListIndex)
	{
		assert(myLineStartingPoints[lineListIndex].size() == myLineEndingPoints[lineListIndex].size());

		const Vec3t<GLfloat>& lineColour = myLineColours[lineListIndex].cast<GLfloat>();

		glColor3f(lineColour[0], lineColour[1], lineColour[2]);
		glLineWidth(GLfloat(myLineWidths[lineListIndex]));

		glBegin(GL_LINES);

		for (int lineIndex = 0; lineIndex < myLineStartingPoints[lineListIndex].size(); ++lineIndex)
		{
			const Vec2t<GLfloat> startPoint = myLineStartingPoints[lineListIndex][lineIndex].cast<GLfloat>();
			glVertex2f(startPoint[0], startPoint[1]);

			const Vec2t<GLfloat> endPoint = myLineEndingPoints[lineListIndex][lineIndex].cast<GLfloat>();
			glVertex2f(endPoint[0], endPoint[1]);			
		}

		glEnd();
	}

	assert(myPoints.size() == myPointColours.size() &&
			myPoints.size() == myPointSizes.size());

	for (int pointListIndex = 0; pointListIndex < myPoints.size(); ++pointListIndex)
	{
		const Vec3t<GLfloat> pointColour = myPointColours[pointListIndex].cast<GLfloat>();
		glColor3f(pointColour[0], pointColour[1], pointColour[2]);
		glPointSize(GLfloat(myPointSizes[pointListIndex]));

		glBegin(GL_POINTS);

		for (int pointIndex = 0; pointIndex < myPoints[pointListIndex].size(); ++pointIndex)
		{
			const Vec2t<GLfloat> point = myPoints[pointListIndex][pointIndex].cast<GLfloat>();
			glVertex2f(point[0], point[1]);
		}

		glEnd();
	}
}

void Renderer::clear()
{
	myPoints.clear();
	myPointColours.clear();
	myPointSizes.clear();

	myLineStartingPoints.clear();
	myLineEndingPoints.clear();
	myLineColours.clear();
	myLineWidths.clear();

	myTriVertices.clear();
	myTriFaces.clear();
	myTriFaceColours.clear();

	myQuadVertices.clear();
	myQuadFaces.clear();
	myQuadFaceColours.clear();
}

void Renderer::run()
{
	glutMainLoop();
}

void Renderer::printImage(const std::string &filename) const
{
	double scale = double(myWindowSize[1]) / myCurrentScreenHeight;
	svg::Point originOffset(-myCurrentScreenOrigin[0], -myCurrentScreenOrigin[1]);

	svg::Document svgDocument(filename + ".svg",
		svg::Layout(svg::Dimensions(myWindowSize[0], myWindowSize[1]),
			svg::Layout::BottomLeft,
			scale, originOffset));

	// Draw rectangles
	assert(myQuadVertices.size() == myQuadFaces.size() &&
			myQuadVertices.size() == myQuadVertices.size());
	for (int quadListIndex = 0; quadListIndex < myQuadFaces.size(); ++quadListIndex)
	{
		assert(myQuadFaces[quadListIndex].size() == myQuadFaceColours[quadListIndex].size());

		for (int quadIndex = 0; quadIndex < myQuadFaces[quadListIndex].size(); ++quadIndex)
		{	
			Vec2d points[4];

			for (int pointIndex : {0, 1, 2, 3})
			{
				int quadVertexIndex = myQuadFaces[quadListIndex][quadIndex][pointIndex];

				assert(quadVertexIndex >= 0 && quadVertexIndex < myQuadVertices[quadListIndex].size());

				points[pointIndex] = myQuadVertices[quadListIndex][quadVertexIndex];
			}

			Vec3i quadColour = (255.f * myQuadFaceColours[quadListIndex][quadIndex]).cast<int>();

			svgDocument << (svg::Polygon(svg::Color(quadColour[0], quadColour[1], quadColour[2]))
								<< svg::Point(points[0][0], points[0][1])
								<< svg::Point(points[1][0], points[1][1])
								<< svg::Point(points[2][0], points[2][1]));

			svgDocument << (svg::Polygon(svg::Color(quadColour[0], quadColour[1], quadColour[2]))
								<< svg::Point(points[0][0], points[0][1])
								<< svg::Point(points[2][0], points[2][1])
								<< svg::Point(points[3][0], points[3][1]));
		}
	}

	// Draw triangles
	assert(myTriVertices.size() == myTriFaces.size() &&
			myTriVertices.size() == myTriFaceColours.size());

	for (int triListIndex = 0; triListIndex < myTriFaces.size(); ++triListIndex)
	{
		assert(myTriFaces[triListIndex].size() == myTriFaceColours[triListIndex].size());

		for (int triIndex = 0; triIndex < myTriFaces[triListIndex].size(); ++triIndex)
		{
			Vec2d points[3];

			for (int pointIndex : {0, 1, 2})
			{
				int triVertexIndex = myTriFaces[triListIndex][triIndex][pointIndex];

				assert(triVertexIndex >= 0 && triVertexIndex < myTriVertices[triListIndex].size());

				points[pointIndex] = myTriVertices[triListIndex][triVertexIndex];
			}
			
			Vec3i triColour = (255.f * myTriFaceColours[triListIndex][triIndex]).cast<int>();

			svgDocument << (svg::Polygon(svg::Color(triColour[0], triColour[1], triColour[2]))
											<< svg::Point(points[0][0], points[0][1])
											<< svg::Point(points[1][0], points[1][1])
											<< svg::Point(points[2][0], points[2][1]));
		}
	}

	// Draw line segments

	assert(myLineStartingPoints.size() == myLineEndingPoints.size() &&
		myLineStartingPoints.size() == myLineColours.size() &&
		myLineStartingPoints.size() == myLineWidths.size());

	for (int lineListIndex = 0; lineListIndex < myLineStartingPoints.size(); ++lineListIndex)
	{
		assert(myLineStartingPoints[lineListIndex].size() == myLineEndingPoints[lineListIndex].size());

		const Vec3i lineColour = (255.f * myLineColours[lineListIndex]).cast<int>();

		for (int lineIndex = 0; lineIndex < myLineStartingPoints[lineListIndex].size(); ++lineIndex)
		{
			Vec2d startPoint = myLineStartingPoints[lineListIndex][lineIndex];
			Vec2d endPoint = myLineEndingPoints[lineListIndex][lineIndex];

			svgDocument << svg::Line(svg::Point(startPoint[0], startPoint[1]),
				svg::Point(endPoint[0], endPoint[1]),
				svg::Stroke(myLineWidths[lineListIndex] / scale, svg::Color(lineColour[0], lineColour[1], lineColour[2])));
		}
	}

	assert(myPoints.size() == myPointColours.size() &&
		myPoints.size() == myPointSizes.size());

	for (int pointListIndex = 0; pointListIndex < myPoints.size(); ++pointListIndex)
	{
		const Vec3i pointColour = (255.f * myPointColours[pointListIndex]).cast<int>();
		double pointSize = myPointSizes[pointListIndex] / scale;

		for (int pointIndex = 0; pointIndex < myPoints[pointListIndex].size(); ++pointIndex)
		{
			const Vec2d& point = myPoints[pointListIndex][pointIndex];

			svgDocument << svg::Circle(svg::Point(point[0], point[1]),
				pointSize,
				svg::Fill(svg::Color(pointColour[0], pointColour[1], pointColour[2])));
		}
	}

	svgDocument.save();
}

}