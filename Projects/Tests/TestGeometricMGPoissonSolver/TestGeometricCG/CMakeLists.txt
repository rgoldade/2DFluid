add_executable(TestGeometricCG TestGeometricCG.cpp)

target_link_libraries(TestGeometricCG
						PRIVATE
						GeometricMGUtilities
						RenderTools
						SurfaceTrackers
						SimTools
						Utilities)

file(RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS TestGeometricCG RUNTIME DESTINATION ${REL})

set_target_properties(TestGeometricCG PROPERTIES FOLDER ${TEST_FOLDER}/${GEOMETRICMG_FOLDER}/TestProjects)