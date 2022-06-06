add_executable(TestQuadtreeGrid TestQuadtreeGrid.cpp)

target_link_libraries(TestQuadtreeGrid
						PRIVATE
						RenderTools
						Utilities)

file(RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS TestQuadtreeGrid RUNTIME DESTINATION ${REL})

set_target_properties(TestQuadtreeGrid PROPERTIES FOLDER ${QUADTREE_FOLDER}/TestProjects)