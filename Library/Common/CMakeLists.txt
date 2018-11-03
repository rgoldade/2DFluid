add_library(2DFluidCommon INTERFACE)

target_include_directories(2DFluidCommon INTERFACE
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

target_link_libraries(2DFluidCommon
						INTERFACE
						2DFluidRenderer)