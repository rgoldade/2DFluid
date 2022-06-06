#include "QuadtreeGrid.h"
#include "Renderer.h"
#include "Utilities.h"

using namespace FluidSim2D;

static std::unique_ptr<QuadtreeGrid> quadtree;
static std::unique_ptr<Renderer> renderer;

static bool isDisplayDirty = true;

void display()
{
	if (isDisplayDirty)
	{
		renderer->clear();

		quadtree->drawGrid(*renderer);
		quadtree->drawCellConnections(*renderer);

		isDisplayDirty = false;

		glutPostRedisplay();
	}
}

void keyboard(unsigned char key, int, int)
{
	if (key == 'r')
	{
		quadtree->refineGrid();
		isDisplayDirty = true;
	}
}

int main(int argc, char** argv)
{
	Vec2i size = Vec2i::Constant(16);
	double dx = PI / double(size[0]);

	// Scene settings
	Transform xform(dx, Vec2d::Zero());

	renderer = std::make_unique<Renderer>("Quadtree test", Vec2i::Constant(1000), xform.offset(), xform.dx() * double(size[0]), &argc, argv);

	quadtree = std::make_unique< QuadtreeGrid>(xform, size, 4);
	int levels = quadtree->levels();
	   
	auto isoSurface = [](const Vec2d& pos) -> double
	{
		return std::min(pos.norm() - .5 * std::sqrt(2.) * PI, (pos - Vec2d::Constant(PI)).norm() - .5 * std::sqrt(2.) * PI);
	};

	auto surfaceRefiner = [&](const Vec2d& pos) -> double
	{
		if (std::fabs(isoSurface(pos)) < dx)
		{
			return 0;
		}

		return -1;
	};

	quadtree->buildTree(surfaceRefiner);

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}