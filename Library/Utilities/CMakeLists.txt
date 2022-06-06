add_library(Utilities
			GridUtilities.h
			Integrator.h
			QuadtreeGrid.cpp
			QuadtreeGrid.h
			QuadtreeVectorInterpolator.cpp
			QuadtreeVectorInterpolator.h
			ScalarGrid.h
			SparseUniformGrid.h
			Timer.h
			Transform.h
			UniformGrid.h
			Utilities.h
			VectorGrid.h)

target_include_directories(Utilities PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

target_link_libraries(Utilities
						PUBLIC
						RenderTools
						TBB::tbb
						Eigen3::Eigen)

set_target_properties(Utilities PROPERTIES FOLDER ${SOURCE_FOLDER})