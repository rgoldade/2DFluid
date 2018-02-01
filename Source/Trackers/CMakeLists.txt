add_library(2dfluid_trackers
				FluidParticles.cpp
				LevelSet2D.cpp
				Mesh2D.cpp
				MeshRebuilder.cpp
				Predicates.cpp)

target_link_libraries(2dfluid_trackers
						PRIVATE
						2dfluid_common)

find_package(EIGEN3 REQUIRED)
if (EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

target_include_directories(2dfluid_trackers PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)