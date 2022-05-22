#include "gtest/gtest.h"

#include "EdgeMesh.h"
#include "InitialGeometry.h"
#include "Utilities.h"

using namespace FluidSim2D;

static void testEdgeMesh(const EdgeMesh& mesh)
{
    const VecVec2d& vertices = mesh.vertices();
    const VecVec2i& edges = mesh.edges();

    std::vector<std::vector<int>> adjacentEdges(vertices.size());
    for (int edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex)
    {
        for (int localVertIndex : {0, 1})
        {
            int vertIndex = edges[edgeIndex][localVertIndex];
            adjacentEdges[vertIndex].push_back(edgeIndex);

            EXPECT_GE(vertIndex, 0);
            EXPECT_LT(vertIndex, vertices.size());
        }
    }

    // Verify that each vertex has two or more adjacent edges. Meaning a
    // closed mesh but allowing for non-manifold meshes.
    for (int vertIndex = 0; vertIndex < vertices.size(); ++vertIndex)
    {
        EXPECT_GE(adjacentEdges[vertIndex].size(), 2);
    }

    // Verify that each vertices' adjacent edge reciprocates
    for (int vertIndex = 0; vertIndex < vertices.size(); ++vertIndex)
    {
        for (int edgeIndex : adjacentEdges[vertIndex])
        {
            EXPECT_TRUE(edges[edgeIndex][0] == vertIndex || edges[edgeIndex][1] == vertIndex);
        }
    }

    // Verify edge's adjacent vertex reciprocates
    for (int edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex)
    {
        for (int localEdgeIndex : {0, 1})
        {
            int vertIndex = edges[edgeIndex][localEdgeIndex];
            EXPECT_TRUE(std::find(adjacentEdges[vertIndex].begin(), adjacentEdges[vertIndex].end(), edgeIndex) != adjacentEdges[vertIndex].end());
        }
    }

    // Bounding box test
    AlignedBox2d meshBBox = mesh.boundingBox();

    for (const Vec2d& vertex : mesh.vertices())
    {
        EXPECT_GE(vertex[0], meshBBox.min()[0]);
        EXPECT_GE(vertex[1], meshBBox.min()[1]);
        
        EXPECT_LE(vertex[0], meshBBox.max()[0]);
        EXPECT_LE(vertex[1], meshBBox.max()[1]);
    }
}

TEST(EDGE_MESH_TESTS, EDGE_MESH_CIRCLE_TEST)
{
    double radius = 1.25;
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);
    testEdgeMesh(mesh);
}

TEST(EDGE_MESH_TESTS, EDGE_MESH_SQUARE_TEST)
{
    Vec2d radius = Vec2d::Random() + 1.5 * Vec2d::Ones();
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeSquareMesh(center, radius);
    testEdgeMesh(mesh);
}

TEST(EDGE_MESH_TESTS, EDGE_MESH_DIAMOND_TEST)
{
    Vec2d radius = Vec2d::Random();
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeDiamondMesh(center, radius);
    testEdgeMesh(mesh);
}

TEST(EDGE_MESH_TESTS, EDGE_COPY_TEST)
{
    double radius = 1.25;
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);
    EdgeMesh copyMesh = mesh;

    EXPECT_EQ(mesh.edgeCount(), copyMesh.edgeCount());
    EXPECT_EQ(mesh.vertexCount(), copyMesh.vertexCount());

    for (int edgeIndex = 0; edgeIndex < mesh.edgeCount(); ++edgeIndex)
    {
        EXPECT_EQ(mesh.edge(edgeIndex)[0], copyMesh.edge(edgeIndex)[0]);
        EXPECT_EQ(mesh.edge(edgeIndex)[1], copyMesh.edge(edgeIndex)[1]);
    }

    for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
    {
        EXPECT_EQ(mesh.vertex(vertIndex)[0], copyMesh.vertex(vertIndex)[0]);
        EXPECT_EQ(mesh.vertex(vertIndex)[1], copyMesh.vertex(vertIndex)[1]);
    }
}

TEST(EDGE_MESH_TESTS, EDGE_REINITIALIZE_TEST)
{
    double radius = 1.25;
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    EdgeMesh reinitMesh = makeCircleMesh(center + Vec2d::Ones(), 2. * radius, 60);

    mesh.reinitialize(reinitMesh.edges(), reinitMesh.vertices());

    EXPECT_EQ(mesh.edgeCount(), reinitMesh.edgeCount());
    EXPECT_EQ(mesh.vertexCount(), reinitMesh.vertexCount());

    for (int edgeIndex = 0; edgeIndex < mesh.edgeCount(); ++edgeIndex)
    {
        EXPECT_EQ(mesh.edge(edgeIndex)[0], reinitMesh.edge(edgeIndex)[0]);
        EXPECT_EQ(mesh.edge(edgeIndex)[1], reinitMesh.edge(edgeIndex)[1]);
    }

    for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
    {
        EXPECT_EQ(mesh.vertex(vertIndex)[0], reinitMesh.vertex(vertIndex)[0]);
        EXPECT_EQ(mesh.vertex(vertIndex)[1], reinitMesh.vertex(vertIndex)[1]);
    }
}

TEST(EDGE_MESH_TESTS, EDGE_MESH_INSERT_TEST)
{
    double radius = 1.25;
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    EdgeMesh copyMesh = mesh;

    EdgeMesh tempMesh = makeSquareMesh(center, Vec2d(radius, radius));

    mesh.insertMesh(tempMesh);

    EXPECT_EQ(mesh.edgeCount(), copyMesh.edgeCount() + tempMesh.edgeCount());
    EXPECT_EQ(mesh.vertexCount(), copyMesh.vertexCount() + tempMesh.vertexCount());

    for (int edgeIndex = 0; edgeIndex < mesh.edgeCount(); ++edgeIndex)
    {
        if (edgeIndex < copyMesh.edgeCount())
        {
            EXPECT_EQ(mesh.edge(edgeIndex)[0], copyMesh.edge(edgeIndex)[0]);
            EXPECT_EQ(mesh.edge(edgeIndex)[1], copyMesh.edge(edgeIndex)[1]);
        }
        else
        {
            Vec2i edge = mesh.edge(edgeIndex) - Vec2i(copyMesh.edgeCount(), copyMesh.edgeCount());
            EXPECT_EQ(edge[0], tempMesh.edge(edgeIndex - copyMesh.edgeCount())[0]);
            EXPECT_EQ(edge[1], tempMesh.edge(edgeIndex - copyMesh.edgeCount())[1]);
        }
    }

    for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
    {
        if (vertIndex < copyMesh.vertexCount())
        {
            EXPECT_EQ(mesh.vertex(vertIndex)[0], copyMesh.vertex(vertIndex)[0]);
            EXPECT_EQ(mesh.vertex(vertIndex)[1], copyMesh.vertex(vertIndex)[1]);
        }
        else
        {
            EXPECT_EQ(mesh.vertex(vertIndex)[0], tempMesh.vertex(vertIndex - copyMesh.vertexCount())[0]);
            EXPECT_EQ(mesh.vertex(vertIndex)[1], tempMesh.vertex(vertIndex - copyMesh.vertexCount())[1]);
        }
    }
}

TEST(EDGE_MESH_TESTS, EDGE_MESH_SET_VERTEX_TEST)
{
    double radius = 1.25;
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    EdgeMesh biggerMesh = makeCircleMesh(center, 1.2 * radius, 40);

    EXPECT_EQ(mesh.edgeCount(), biggerMesh.edgeCount());
    EXPECT_EQ(mesh.vertexCount(), biggerMesh.vertexCount());

    for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
    {
        mesh.setVertex(vertIndex, biggerMesh.vertex(vertIndex));
    }

    for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
    {
        EXPECT_EQ(mesh.vertex(vertIndex)[0], biggerMesh.vertex(vertIndex)[0]);
        EXPECT_EQ(mesh.vertex(vertIndex)[1], biggerMesh.vertex(vertIndex)[1]);
    }
}

TEST(EDGE_MESH_TESTS, EDGE_MESH_OFFSET_TEST)
{
    double radius = 1.25;
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    EdgeMesh biggerMesh = makeCircleMesh(center, 1.2 * radius, 40);

    EXPECT_EQ(mesh.edgeCount(), biggerMesh.edgeCount());
    EXPECT_EQ(mesh.vertexCount(), biggerMesh.vertexCount());

    for (int edgeIndex = 0; edgeIndex < mesh.edgeCount(); ++edgeIndex)
    {
        const Vec2i& edge = mesh.edge(edgeIndex);
        Vec2d midPoint = .5 * (mesh.vertex(edge[0]) + mesh.vertex(edge[1]));

        Vec2d biggerMeshMidPoint = .5 * (biggerMesh.vertex(edge[0]) + biggerMesh.vertex(edge[1]));

        Vec2d offsetNorm = (biggerMeshMidPoint - midPoint) / (biggerMeshMidPoint - midPoint).norm();

        Vec2d meshNorm = mesh.normal(edgeIndex);
        Vec2d biggerMeshNorm = biggerMesh.normal(edgeIndex);

        EXPECT_TRUE(isNearlyEqual(offsetNorm[0], meshNorm[0]));
        EXPECT_TRUE(isNearlyEqual(offsetNorm[1], meshNorm[1]));

        EXPECT_TRUE(isNearlyEqual(offsetNorm[0], biggerMeshNorm[0]));
        EXPECT_TRUE(isNearlyEqual(offsetNorm[1], biggerMeshNorm[1]));
    }
}

TEST(EDGE_MESH_TESTS, EDGE_MESH_REVERSE_TEST)
{
    double radius = 1;
    Vec2d center = Vec2d::Zero();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    EdgeMesh reverseMesh = mesh;
    reverseMesh.reverse();

    EXPECT_EQ(mesh.edgeCount(), reverseMesh.edgeCount());
    EXPECT_EQ(mesh.vertexCount(), reverseMesh.vertexCount());

    for (int edgeIndex = 0; edgeIndex < mesh.edgeCount(); ++edgeIndex)
    {
        double normDot = mesh.normal(edgeIndex).dot(reverseMesh.normal(edgeIndex));
        EXPECT_TRUE(isNearlyEqual(normDot, -1.));
    }
}

TEST(EDGE_MESH_TESTS, EDGE_MESH_NORMAL_TEST)
{
    double radius = 1.25;
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    for (int edgeIndex = 0; edgeIndex < mesh.edgeCount(); ++edgeIndex)
    {
        const Vec2i& edge = mesh.edge(edgeIndex);
        Vec2d midPoint = .5 * (mesh.vertex(edge[0]) + mesh.vertex(edge[1]));

        Vec2d meshNormal = mesh.normal(edgeIndex);

        double denom = std::sqrt(std::pow(midPoint[0] - center[0], 2) + std::pow(midPoint[1] - center[1], 2));
        Vec2d pointNormal = Vec2d((midPoint[0] - center[0]) / denom, (midPoint[1] - center[1]) / denom);
        pointNormal /= pointNormal.norm();

        EXPECT_TRUE(isNearlyEqual(pointNormal[0], meshNormal[0]));
        EXPECT_TRUE(isNearlyEqual(pointNormal[1], meshNormal[1]));
    }
}

TEST(EDGE_MESH_TESTS, EDGE_DEGENERATE_TEST)
{
    double radius = 1.25;
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    for (int edgeIndex = 0; edgeIndex < mesh.edgeCount(); ++edgeIndex)
    {
        EXPECT_FALSE(mesh.isEdgeDegenerate(edgeIndex));
    }

    for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
    {
        mesh.setVertex(vertIndex, center);
    }

    for (int edgeIndex = 0; edgeIndex < mesh.edgeCount(); ++edgeIndex)
    {
        EXPECT_TRUE(mesh.isEdgeDegenerate(edgeIndex));
    }
}

TEST(EDGE_MESH_TESTS, EDGE_ADVECT_TEST)
{
    double radius = 1.25;
    Vec2d center = Vec2d::Random();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    EdgeMesh copyMesh = mesh;

    auto velocity = [&](const double, const Vec2d& point) -> Vec2d { return point - center; };

    mesh.advectMesh(1., velocity, IntegrationOrder::FORWARDEULER);

    for (int vertIndex = 0; vertIndex < copyMesh.vertexCount(); ++vertIndex)
    {
        Vec2d copyVertex = copyMesh.vertex(vertIndex) + velocity(0., copyMesh.vertex(vertIndex));
        Vec2d vertex = mesh.vertex(vertIndex);
        EXPECT_TRUE(isNearlyEqual(vertex[0], copyVertex[0]));
        EXPECT_TRUE(isNearlyEqual(vertex[1], copyVertex[1]));
    }
}