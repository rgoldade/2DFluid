add_library(SurfaceTrackers
				EdgeMesh.cpp
				FluidParticles.cpp
				InitialGeometry.cpp
				LevelSet.cpp
				Predicates.cpp)

target_link_libraries(SurfaceTrackers
						PRIVATE
						RenderTools
						SimTools
						Utilities)

target_include_directories(SurfaceTrackers PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(SurfaceTrackers PROPERTIES FOLDER ${SOURCE_FOLDER})