cmake_minimum_required(VERSION 3.0)
set (CMAKE_CXX_STANDARD 11)
project(2DFluid)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

find_package(OpenGL REQUIRED)
if(MSVC)
	find_package(FREEGLUT REQUIRED)
else()
	find_package(GLUT REQUIRED)
endif()

if(MSVC)
	include_directories( ${OPENGL_INCLUDE_DIR}  ${FREEGLUT_INCLUDE_DIR} )
	link_libraries(${OPENGL_LIBRARY} ${FREEGLUT_LIBRARY} )
else()
	include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
	link_libraries(${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
endif()

enable_testing()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


add_subdirectory(Source)
add_subdirectory(Projects)

