add_executable(LevelSetLiquid LevelSetLiquid.cpp)

target_link_libraries(LevelSetLiquid 
						PRIVATE
						RegularGridSimLibrary
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS LevelSetLiquid RUNTIME DESTINATION ${REL})

set_target_properties(LevelSetLiquid PROPERTIES FOLDER ${REGULAR_FOLDER})
