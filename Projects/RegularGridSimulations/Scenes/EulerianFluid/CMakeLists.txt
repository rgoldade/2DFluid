add_executable(eulerianfluid eulerianfluid.cpp)

target_link_libraries(eulerianfluid
						PRIVATE
						2dfluid_library
						2dfluid_trackers
						2dfluid_simtools
						2dfluid_common
						2dfluid_render)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS eulerianfluid RUNTIME DESTINATION ${REL})

set_target_properties(eulerianfluid PROPERTIES FOLDER ${REGULAR_FOLDER})
