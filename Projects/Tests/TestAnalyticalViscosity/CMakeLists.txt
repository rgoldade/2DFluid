add_executable(TestAnalyticalViscosity AnalyticalViscositySolver.cpp TestAnalyticalViscosity.cpp )

target_link_libraries(TestAnalyticalViscosity
						PRIVATE
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS TestAnalyticalViscosity RUNTIME DESTINATION ${REL})

set_target_properties(TestAnalyticalViscosity PROPERTIES FOLDER ${TEST_FOLDER})