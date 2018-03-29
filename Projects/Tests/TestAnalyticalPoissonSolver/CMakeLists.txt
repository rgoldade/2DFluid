add_executable(testanalyticalpoissonsolver AnalyticalPoissonSolver.cpp testanalyticalpoissonsolver.cpp )

target_link_libraries(testanalyticalpoissonsolver
						PRIVATE
						2dfluid_trackers
						2dfluid_simtools
						2dfluid_common
						2dfluid_render)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS testanalyticalpoissonsolver RUNTIME DESTINATION ${REL})

set_target_properties(testanalyticalpoissonsolver PROPERTIES FOLDER ${TEST_FOLDER})
