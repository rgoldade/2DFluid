add_executable(ViscousLiquid ViscousLiquid.cpp)

target_link_libraries(ViscousLiquid 
						PRIVATE
						2DFluidLibrary
						2DFluidTrackers
						2DFluidSimTools
						2DFluidCommon
						2DFluidRenderer)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS ViscousLiquid RUNTIME DESTINATION ${REL})

set_target_properties(ViscousLiquid PROPERTIES FOLDER ${REGULAR_FOLDER})
