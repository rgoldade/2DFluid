add_library(2DFluidRenderer Renderer.cpp)

target_include_directories(2DFluidRenderer  PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(2DFluidRenderer PROPERTIES FOLDER ${SOURCE_FOLDER})