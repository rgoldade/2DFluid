cmake_minimum_required(VERSION 3.9)
project(2DFluid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

find_package(OpenGL REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(MSVC)
	find_package(FREEGLUT REQUIRED)
else()
	find_package(GLUT REQUIRED)
endif()

find_package(TBB CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

if(MSVC)
	include_directories( ${OPENGL_INCLUDE_DIR}  ${FREEGLUT_INCLUDE_DIR} )
	link_libraries(${OPENGL_LIBRARY} ${FREEGLUT_LIBRARY} )
else()
	include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
	link_libraries(${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(Library)
add_subdirectory(Projects)

enable_testing()
add_subdirectory(UnitTests)
