cmake_minimum_required(VERSION 3.9)
set (CMAKE_CXX_STANDARD 17)
project(2DFluid)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# When using Homebrew LLVM, the compiled objects require Homebrew's libc++ rather
# than the older macOS system libc++. Point the linker at it explicitly.
if(CMAKE_CXX_COMPILER MATCHES "/opt/homebrew/opt/llvm")
    set(HOMEBREW_LLVM_LIBCXX_DIR "/opt/homebrew/opt/llvm/lib/c++")
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -L${HOMEBREW_LLVM_LIBCXX_DIR} -Wl,-rpath,${HOMEBREW_LLVM_LIBCXX_DIR}")
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -L${HOMEBREW_LLVM_LIBCXX_DIR} -Wl,-rpath,${HOMEBREW_LLVM_LIBCXX_DIR}")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(EIGEN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/externals/eigen)
include_directories(${EIGEN_INCLUDE_DIR})

set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TBB_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory(externals/onetbb)
link_libraries(TBB::tbb)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(externals/polyscope)

add_subdirectory(Library)
add_subdirectory(Projects)

enable_testing()
add_subdirectory(UnitTests)
