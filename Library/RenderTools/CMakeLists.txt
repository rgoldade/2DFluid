add_library(RenderTools Renderer.cpp)

target_include_directories(RenderTools  PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(RenderTools PROPERTIES FOLDER ${SOURCE_FOLDER})