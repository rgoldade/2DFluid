add_executable(euleriansmoke euleriansmoke.cpp)

target_link_libraries(euleriansmoke 
						PRIVATE
						2dfluid_library
						2dfluid_trackers
						2dfluid_simtools
						2dfluid_common
						2dfluid_render)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS euleriansmoke RUNTIME DESTINATION ${REL})

set_target_properties(euleriansmoke PROPERTIES FOLDER ${REGULAR_FOLDER})
