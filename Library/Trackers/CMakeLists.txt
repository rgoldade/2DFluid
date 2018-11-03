add_library(2DFluidTrackers
				FluidParticles.cpp
				LevelSet2D.cpp
				Mesh2D.cpp
				Predicates.cpp)

target_link_libraries(2DFluidTrackers
						PRIVATE
						2DFluidCommon
						2DFluidSimTools)

find_package(EIGEN3 REQUIRED)
if (EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

target_include_directories(2DFluidTrackers PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(2DFluidTrackers PROPERTIES FOLDER ${SOURCE_FOLDER})