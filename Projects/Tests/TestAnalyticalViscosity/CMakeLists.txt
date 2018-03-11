add_executable(testanalyticalviscosity AnalyticalViscositySolver.cpp testanalyticalviscosity.cpp )

target_link_libraries(testanalyticalviscosity
						PRIVATE
						2dfluid_trackers
						2dfluid_simtools
						2dfluid_common
						2dfluid_render)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS testanalyticalviscosity RUNTIME DESTINATION ${REL})

set_target_properties(testanalyticalviscosity PROPERTIES FOLDER ${TEST_FOLDER})