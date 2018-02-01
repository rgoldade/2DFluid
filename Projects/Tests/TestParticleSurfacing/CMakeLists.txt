add_executable(testparticlesurfacing testparticlesurfacing.cpp)

target_link_libraries(testparticlesurfacing
						PRIVATE
						2dfluid_trackers
						2dfluid_simtools
						2dfluid_common
						2dfluid_render)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS testparticlesurfacing RUNTIME DESTINATION ${REL})
