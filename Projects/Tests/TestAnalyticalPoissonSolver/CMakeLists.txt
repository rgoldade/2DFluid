add_executable(TestAnalyticalPoissonSolver AnalyticalPoissonSolver.cpp TestAnalyticalPoissonSolver.cpp )

target_link_libraries(TestAnalyticalPoissonSolver
						PRIVATE
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS TestAnalyticalPoissonSolver RUNTIME DESTINATION ${REL})

set_target_properties(TestAnalyticalPoissonSolver PROPERTIES FOLDER ${TEST_FOLDER})
