add_executable(TestSymmetry TestSymmetry.cpp)

target_link_libraries(TestSymmetry
						PRIVATE
						GeometricMGUtilities
						RenderTools
						SurfaceTrackers
						SimTools
						Utilities)

file(RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS TestSymmetry RUNTIME DESTINATION ${REL})

set_target_properties(TestSymmetry PROPERTIES FOLDER ${TEST_FOLDER}/${GEOMETRICMG_FOLDER}/TestProjects)