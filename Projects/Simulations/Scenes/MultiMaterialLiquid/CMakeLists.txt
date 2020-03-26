add_executable(MultiMaterialBubbles MultiMaterialBubbles.cpp)

target_link_libraries(MultiMaterialBubbles
						PRIVATE
						RegularGridSimLibrary
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS MultiMaterialBubbles RUNTIME DESTINATION ${REL})

set_target_properties(MultiMaterialBubbles PROPERTIES FOLDER ${REGULAR_FOLDER})

add_executable(MultiMaterialLayers MultiMaterialLayers.cpp)

target_link_libraries(MultiMaterialLayers
						PRIVATE
						RegularGridSimLibrary
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS MultiMaterialLayers RUNTIME DESTINATION ${REL})

set_target_properties(MultiMaterialLayers PROPERTIES FOLDER ${REGULAR_FOLDER})