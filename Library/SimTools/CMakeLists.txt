add_library(SimTools
				AdaptiveViscositySolver.cpp
				ComputeWeights.cpp
				GeometricMultigridOperators.cpp
				GeometricMultigridPoissonSolver.cpp
				GeometricPressureProjection.cpp
				Noise.cpp
				PressureProjection.cpp
				ViscositySolver.cpp)

target_link_libraries(SimTools
						PRIVATE
						SurfaceTrackers
						Utilities)

target_include_directories(SimTools PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(SimTools PROPERTIES FOLDER ${SOURCE_FOLDER})
