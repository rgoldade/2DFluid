cmake_minimum_required(VERSION 3.0)
set (CMAKE_CXX_STANDARD 11)
project(2DFluid)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

enable_testing()

add_subdirectory(Source)
add_subdirectory(Projects)

