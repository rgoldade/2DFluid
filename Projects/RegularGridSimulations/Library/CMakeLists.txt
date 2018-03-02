add_library(2dfluid_library EulerianLiquid.cpp EulerianSmoke.cpp)

target_link_libraries(2dfluid_library
						PRIVATE
						2dfluid_trackers
						2dfluid_simtools
						2dfluid_common
						2dfluid_render)

target_include_directories(2dfluid_library PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(2dfluid_library PROPERTIES FOLDER ${REGULAR_FOLDER}/Library)
