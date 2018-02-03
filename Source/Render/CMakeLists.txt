add_library(2dfluid_render Renderer.cpp)

target_include_directories(2dfluid_render PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)
