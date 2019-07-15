add_library(2DGeometricMGUtilities INTERFACE)

target_include_directories(2DGeometricMGUtilities INTERFACE
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

target_link_libraries(2DGeometricMGUtilities
						INTERFACE
						2DFluidCommon
						2DFluidSimTools)