add_library(GeometricMGUtilities INTERFACE)

target_include_directories(GeometricMGUtilities INTERFACE
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

target_link_libraries(GeometricMGUtilities
						INTERFACE
						SimTools
						Utilities)