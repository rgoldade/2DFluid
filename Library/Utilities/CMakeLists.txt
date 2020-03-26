add_library(Utilities INTERFACE)

target_include_directories(Utilities INTERFACE
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

target_link_libraries(Utilities
						INTERFACE
						RenderTools)