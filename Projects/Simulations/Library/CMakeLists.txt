add_library(RegularGridSimLibrary EulerianLiquidSimulator.cpp EulerianSmokeSimulator.cpp MultiMaterialLiquidSimulator.cpp MultiMaterialPressureProjection.cpp)

target_link_libraries(RegularGridSimLibrary
						PRIVATE
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

target_include_directories(RegularGridSimLibrary PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(RegularGridSimLibrary PROPERTIES FOLDER ${REGULAR_FOLDER}/Library)
