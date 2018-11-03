add_library(2DFluidLibrary EulerianLiquid.cpp EulerianSmoke.cpp)

target_link_libraries(2DFluidLibrary
						PRIVATE
						2DFluidTrackers
						2DFluidSimTools
						2DFluidCommon
						2DFluidRenderer)

target_include_directories(2DFluidLibrary PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(2DFluidLibrary PROPERTIES FOLDER ${REGULAR_FOLDER}/Library)
