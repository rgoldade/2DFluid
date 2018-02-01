add_library(2dfluid_common
				OctreeGrid.cpp)

target_include_directories(2dfluid_common PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

target_link_libraries(2dfluid_common
						PRIVATE
						2dfluid_render)
