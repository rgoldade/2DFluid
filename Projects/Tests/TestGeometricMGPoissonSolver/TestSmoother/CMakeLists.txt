add_executable(TestMGSmoother TestSmoother.cpp)

target_link_libraries(TestMGSmoother
						PRIVATE
						GeometricMGUtilities
						RenderTools
						SurfaceTrackers
						SimTools
						Utilities)

file(RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS TestMGSmoother RUNTIME DESTINATION ${REL})

set_target_properties(TestMGSmoother PROPERTIES FOLDER ${TEST_FOLDER}/${GEOMETRICMG_FOLDER}/TestProjects)