add_executable(TestParticleSurfacing TestParticleSurfacing.cpp)

target_link_libraries(TestParticleSurfacing PRIVATE
						2DFluidTrackers
						2DFluidSimTools
						2DFluidCommon
						2DFluidRenderer)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS TestParticleSurfacing RUNTIME DESTINATION ${REL})

set_target_properties(TestParticleSurfacing PROPERTIES FOLDER ${TEST_FOLDER})