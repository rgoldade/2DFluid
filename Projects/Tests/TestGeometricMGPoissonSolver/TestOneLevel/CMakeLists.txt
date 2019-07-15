add_executable(TestOneLevel TestOneLevel.cpp)

target_link_libraries(TestOneLevel
						PRIVATE
						2DGeometricMGUtilities
						2DFluidLibrary
						2DFluidTrackers
						2DFluidSimTools
						2DFluidCommon
						2DFluidRenderer)

file(RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS TestOneLevel RUNTIME DESTINATION ${REL})

set_target_properties(TestOneLevel PROPERTIES FOLDER ${TEST_FOLDER}/${GEOMETRICMG_FOLDER}/TestProjects)