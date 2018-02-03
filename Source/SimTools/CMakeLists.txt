add_library(2dfluid_simtools
				AdvectField.cpp
				ComputeWeights.cpp
				ExtrapolateField.cpp
				Noise.cpp
				PressureProjection.cpp
				ViscositySolver.cpp)

target_link_libraries(2dfluid_simtools
						PRIVATE
						2dfluid_common
						2dfluid_trackers)

find_package(EIGEN3 REQUIRED)
if (EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

target_include_directories(2dfluid_simtools PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(2dfluid_simtools PROPERTIES FOLDER ${SOURCE_FOLDER})