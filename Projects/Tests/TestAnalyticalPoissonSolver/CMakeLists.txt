add_executable(TestAnalyticalPoissonSolver AnalyticalPoissonSolver.cpp TestAnalyticalPoissonSolver.cpp )

target_link_libraries(TestAnalyticalPoissonSolver
						PRIVATE
						2DFluidTrackers
						2DFluidSimTools
						2DFluidCommon
						2DFluidRenderer)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS TestAnalyticalPoissonSolver RUNTIME DESTINATION ${REL})

set_target_properties(TestAnalyticalPoissonSolver PROPERTIES FOLDER ${TEST_FOLDER})
