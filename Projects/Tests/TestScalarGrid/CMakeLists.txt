add_executable(testscalargrid testscalargrid.cpp)

target_link_libraries(testscalargrid
						PRIVATE
						2dfluid_trackers
						2dfluid_common
						2dfluid_render)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS testscalargrid RUNTIME DESTINATION ${REL})

set_target_properties(testscalargrid PROPERTIES FOLDER ${TEST_FOLDER})