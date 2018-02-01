add_executable(testlevelset testlevelset.cpp)

target_link_libraries(testlevelset
						PRIVATE
						2dfluid_trackers
						2dfluid_common
						2dfluid_render)

file(RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS testlevelset RUNTIME DESTINATION ${REL})
