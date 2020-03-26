add_executable(ViscousLiquid ViscousLiquid.cpp)

target_link_libraries(ViscousLiquid 
						PRIVATE
						RegularGridSimLibrary
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS ViscousLiquid RUNTIME DESTINATION ${REL})

set_target_properties(ViscousLiquid PROPERTIES FOLDER ${REGULAR_FOLDER})
