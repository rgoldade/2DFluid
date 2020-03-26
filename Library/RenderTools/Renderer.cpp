#include "Renderer.h"

#include "simple_svg_1.0.0.hpp"

namespace FluidSim2D::RenderTools
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

Renderer::Renderer(const char *title, Vec2i windowSize, Vec2f screenOrigin,
	float screenHeight, int *argc, char **argv)
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
	glOrtho(myCurrentScreenOrigin[0], myCurrentScreenOrigin[0] + (myCurrentScreenHeight * float(myWindowSize[0])) / float(myWindowSize[1]), myCurrentScreenOrigin[1], myCurrentScreenOrigin[1] + myCurrentScreenHeight, 0, 1);
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
			float scale;
			switch (myMouseAction)
			{
				// PAN is only for drag
				// TODO: add click specific movement
			case MouseAction::ZOOM_IN:

				// Zoom in by 2x
				scale = .5;

				myCurrentScreenOrigin[0] -= .5 * (scale - 1.) * float(myWindowSize[0]) * myCurrentScreenHeight / float(myWindowSize[1]);
				myCurrentScreenOrigin[1] -= .5 * (scale - 1.) * myCurrentScreenHeight;
				myCurrentScreenHeight *= scale;

				glutPostRedisplay();
				break;

			case MouseAction::ZOOM_OUT:

				scale = 2.;
				myCurrentScreenOrigin[0] -= .5 * (scale - 1.) * float(myWindowSize[0]) * myCurrentScreenHeight / float(myWindowSize[1]);
				myCurrentScreenOrigin[1] -= .5 * (scale - 1.) * myCurrentScreenHeight;
				myCurrentScreenHeight *= scale;

				glutPostRedisplay();
			}
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
				float pixelRatio = myCurrentScreenHeight / float(myWindowSize[1]);
				myCurrentScreenOrigin[0] -= pixelRatio * (float(x) - myMousePosition[0]);
				myCurrentScreenOrigin[1] += pixelRatio * (float(y) - myMousePosition[1]);
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
void Renderer::addPoint(const Vec2f& point, const Vec3f& colour, float size)
{
	myPoints.emplace_back(1, point);
	myPointColours.push_back(colour);
	myPointSizes.push_back(size);
}

void Renderer::addPoints(const std::vector<Vec2f>& points, const Vec3f& colour, float size)
{
	myPoints.push_back(points);
	myPointColours.push_back(colour);
	myPointSizes.push_back(size);
}

void Renderer::addLine(const Vec2f& startingPoint, const Vec2f& endingPoint, const Vec3f& colour, float lineWidth)
{
	myLineStartingPoints.emplace_back(1, startingPoint);
	myLineEndingPoints.emplace_back(1, endingPoint);

	myLineColours.push_back(colour);
	myLineWidths.push_back(lineWidth);
}

void Renderer::addLines(const std::vector<Vec2f>& startingPoints, const std::vector<Vec2f>& endingPoints, const Vec3f& colour, float lineWidth)
{
	assert(startingPoints.size() == endingPoints.size());
	myLineStartingPoints.push_back(startingPoints);
	myLineEndingPoints.push_back(endingPoints);

	myLineColours.push_back(colour);
	myLineWidths.push_back(lineWidth);
}

void Renderer::addTriFaces(const std::vector<Vec2f>& vertices, const std::vector<Vec3i>& faces, const std::vector<Vec3f>& faceColours)
{
	myTriVertices.push_back(vertices);
	myTriFaces.push_back(faces);
	myTriFaceColours.push_back(faceColours);
}

void Renderer::addQuadFaces(const std::vector<Vec2f>& vertices, const std::vector<Vec4i>& faces, const std::vector<Vec3f>& faceColours)
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
			const Vec3f& quadFaceColour = myQuadFaceColours[quadListIndex][quadIndex];
			glColor3f(quadFaceColour[0], quadFaceColour[1], quadFaceColour[2]);

			glBegin(GL_QUADS);

			for (int pointIndex : {0, 1, 2, 3})
			{
				int quadVertexIndex = myQuadFaces[quadListIndex][quadIndex][pointIndex];

				assert(quadVertexIndex >= 0 && quadVertexIndex < myQuadVertices[quadListIndex].size());

				const Vec2f& vertexPoint = myQuadVertices[quadListIndex][quadVertexIndex];
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
			const Vec3f& triFaceColour = myTriFaceColours[triListIndex][triIndex];
			glColor3f(triFaceColour[0], triFaceColour[1], triFaceColour[2]);

			glBegin(GL_TRIANGLES);

			for (int pointIndex : {0, 1, 2})
			{
				int triVertexIndex = myTriFaces[triListIndex][triIndex][pointIndex];

				assert(triVertexIndex >= 0 && triVertexIndex < myTriVertices[triListIndex].size());

				const Vec2f& vertexPoint = myTriVertices[triListIndex][triVertexIndex];
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

		const Vec3f& lineColour = myLineColours[lineListIndex];

		glColor3f(lineColour[0], lineColour[1], lineColour[2]);
		glLineWidth(myLineWidths[lineListIndex]);

		glBegin(GL_LINES);

		for (int lineIndex = 0; lineIndex < myLineStartingPoints[lineListIndex].size(); ++lineIndex)
		{
			const Vec2f& startPoint = myLineStartingPoints[lineListIndex][lineIndex];
			glVertex2f(startPoint[0], startPoint[1]);

			const Vec2f& endPoint = myLineEndingPoints[lineListIndex][lineIndex];
			glVertex2f(endPoint[0], endPoint[1]);			
		}

		glEnd();
	}

	assert(myPoints.size() == myPointColours.size() &&
			myPoints.size() == myPointSizes.size());

	for (int pointListIndex = 0; pointListIndex < myPoints.size(); ++pointListIndex)
	{
		const Vec3f& pointColour = myPointColours[pointListIndex];
		glColor3f(pointColour[0], pointColour[1], pointColour[2]);
		glPointSize(myPointSizes[pointListIndex]);

		glBegin(GL_POINTS);

		for (int pointIndex = 0; pointIndex < myPoints[pointListIndex].size(); ++pointIndex)
		{
			const Vec2f& point = myPoints[pointListIndex][pointIndex];
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
	float scale = float(myWindowSize[1]) / myCurrentScreenHeight;
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
			Vec3f quadColour = 255 * myQuadFaceColours[quadListIndex][quadIndex];
	
			Vec2f points[4];

			for (int pointIndex : {0, 1, 2, 3})
			{
				int quadVertexIndex = myQuadFaces[quadListIndex][quadIndex][pointIndex];

				assert(quadVertexIndex >= 0 && quadVertexIndex < myQuadVertices[quadListIndex].size());

				points[pointIndex] = myQuadVertices[quadListIndex][quadVertexIndex];
			}

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
			Vec2f points[3];

			Vec3f triColour = 255 * myTriFaceColours[triListIndex][triIndex];

			for (int pointIndex : {0, 1, 2})
			{
				int triVertexIndex = myTriFaces[triListIndex][triIndex][pointIndex];

				assert(triVertexIndex >= 0 && triVertexIndex < myTriVertices[triListIndex].size());

				points[pointIndex] = myTriVertices[triListIndex][triVertexIndex];
			}

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

		const Vec3f& lineColour = 255 * myLineColours[lineListIndex];

		for (int lineIndex = 0; lineIndex < myLineStartingPoints[lineListIndex].size(); ++lineIndex)
		{
			Vec2f startPoint = myLineStartingPoints[lineListIndex][lineIndex];
			Vec2f endPoint = myLineEndingPoints[lineListIndex][lineIndex];

			svgDocument << svg::Line(svg::Point(startPoint[0], startPoint[1]),
				svg::Point(endPoint[0], endPoint[1]),
				svg::Stroke(myLineWidths[lineListIndex] / scale, svg::Color(lineColour[0], lineColour[1], lineColour[2])));
		}
	}

	assert(myPoints.size() == myPointColours.size() &&
		myPoints.size() == myPointSizes.size());

	for (int pointListIndex = 0; pointListIndex < myPoints.size(); ++pointListIndex)
	{
		const Vec3f& pointColour = 255 * myPointColours[pointListIndex];
		float pointSize = myPointSizes[pointListIndex] / scale;

		for (int pointIndex = 0; pointIndex < myPoints[pointListIndex].size(); ++pointIndex)
		{
			const Vec2f& point = myPoints[pointListIndex][pointIndex];

			svgDocument << svg::Circle(svg::Point(point[0], point[1]),
				pointSize,
				svg::Fill(svg::Color(pointColour[0], pointColour[1], pointColour[2])));
		}
	}

	svgDocument.save();
}

}