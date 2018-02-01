add_executable(testmovingmeshandlevelset testmovingmeshandlevelset.cpp)

target_link_libraries(testmovingmeshandlevelset
						PRIVATE
						2dfluid_trackers
						2dfluid_simtools
						2dfluid_common
						2dfluid_render)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS testmovingmeshandlevelset RUNTIME DESTINATION ${REL})
