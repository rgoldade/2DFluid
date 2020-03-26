add_executable(TestScalarGrid TestScalarGrid.cpp)

target_link_libraries(TestScalarGrid PRIVATE
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS TestScalarGrid RUNTIME DESTINATION ${REL})

set_target_properties(TestScalarGrid PROPERTIES FOLDER ${TEST_FOLDER})