cmake_minimum_required(VERSION 3.9)
set (CMAKE_CXX_STANDARD 14)
project(2DFluid)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

find_package(OpenGL REQUIRED)

find_package(OpenMP REQUIRED)
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

if(MSVC)
	include_directories( ${OPENGL_INCLUDE_DIR}  ${FREEGLUT_INCLUDE_DIR} )
	link_libraries(${OPENGL_LIBRARY} ${FREEGLUT_LIBRARY} )
else()
	include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
	link_libraries(${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
endif()

find_package(EIGEN3 REQUIRED)
if (EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

find_package(TBB REQUIRED)
if (TBB_FOUND)
	include_directories(${TBB_INCLUDE_DIRS})
	link_libraries(${TBB_LIBRARIES} ${TBB_LIBRARIES_DEBUG})
endif()

enable_testing()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


add_subdirectory(Library)
add_subdirectory(Projects)

