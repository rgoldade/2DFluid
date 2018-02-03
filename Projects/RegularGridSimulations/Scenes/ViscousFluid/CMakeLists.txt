add_executable(viscousfluid viscousfluid.cpp)

target_link_libraries(viscousfluid
						PRIVATE
						2dfluid_simulators
						2dfluid_trackers
						2dfluid_simtools
						2dfluid_common
						2dfluid_render)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS viscousfluid RUNTIME DESTINATION ${REL})

set_target_properties(viscousfluid PROPERTIES FOLDER ${REGULAR_FOLDER})