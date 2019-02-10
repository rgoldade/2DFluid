add_executable(RisingSmoke RisingSmoke.cpp)

target_link_libraries(RisingSmoke 
						PRIVATE
						2DFluidLibrary
						2DFluidTrackers
						2DFluidSimTools
						2DFluidCommon
						2DFluidRenderer)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS RisingSmoke RUNTIME DESTINATION ${REL})

set_target_properties(RisingSmoke PROPERTIES FOLDER ${REGULAR_FOLDER})
