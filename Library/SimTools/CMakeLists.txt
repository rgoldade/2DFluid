add_library(2DFluidSimTools
				ComputeWeights.cpp
				GeometricMGOperations.cpp
				GeometricMGPoissonSolver.cpp
				GeometricPressureProjection.cpp
				PressureProjection.cpp
				ViscositySolver.cpp
				Noise.cpp)

target_link_libraries(2DFluidSimTools
						PRIVATE
						2DFluidCommon
						2DFluidTrackers)

target_include_directories(2DFluidSimTools PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(2DFluidSimTools PROPERTIES FOLDER ${SOURCE_FOLDER})
