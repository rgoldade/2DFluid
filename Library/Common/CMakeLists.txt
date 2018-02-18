add_library(2dfluid_common INTERFACE)

target_include_directories(2dfluid_common INTERFACE
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

target_link_libraries(2dfluid_common
						INTERFACE
						2dfluid_render)