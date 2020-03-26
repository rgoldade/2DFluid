add_executable(TestLevelSet TestLevelSet.cpp)

target_link_libraries(TestLevelSet 
						PRIVATE
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file(RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS TestLevelSet RUNTIME DESTINATION ${REL})

set_target_properties(TestLevelSet PROPERTIES FOLDER ${TEST_FOLDER})