add_executable(MultiMaterialLiquids MultiMaterialBubbles.cpp)

target_link_libraries(MultiMaterialLiquids
						PRIVATE
						2DFluidLibrary
						2DFluidTrackers
						2DFluidSimTools
						2DFluidCommon
						2DFluidRenderer)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS MultiMaterialLiquids RUNTIME DESTINATION ${REL})

set_target_properties(MultiMaterialLiquids PROPERTIES FOLDER ${REGULAR_FOLDER})
