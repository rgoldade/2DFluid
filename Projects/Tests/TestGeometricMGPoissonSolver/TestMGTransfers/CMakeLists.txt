add_executable(TestMGTransfers TestMGTransfers.cpp)

target_link_libraries(TestMGTransfers
						PRIVATE
						GeometricMGUtilities
						RenderTools
						SurfaceTrackers
						SimTools
						Utilities)

file(RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS TestMGTransfers RUNTIME DESTINATION ${REL})

set_target_properties(TestMGTransfers PROPERTIES FOLDER ${TEST_FOLDER}/${GEOMETRICMG_FOLDER}/TestProjects)