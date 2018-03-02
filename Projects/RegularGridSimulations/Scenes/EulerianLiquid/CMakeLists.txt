add_executable(eulerianliquid eulerianliquid.cpp)

target_link_libraries(eulerianliquid 
						PRIVATE
						2dfluid_library
						2dfluid_trackers
						2dfluid_simtools
						2dfluid_common
						2dfluid_render)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS eulerianliquid RUNTIME DESTINATION ${REL})

set_target_properties(eulerianliquid PROPERTIES FOLDER ${REGULAR_FOLDER})
