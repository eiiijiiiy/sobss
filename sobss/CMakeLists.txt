project(sobss)

cmake_minimum_required(VERSION 3.18)
set (CMAKE_CXX_STANDARD 17)

find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/cmake)
find_package(RapidJSON)
find_package(Eigen3 REQUIRED)
find_package(xtensor REQUIRED)
find_package(NLopt HINTS ${CMAKE_INSTALL_PREFIX}/lib/cmake)
find_package(CGAL COMPONENTS Core REQUIRED)

if (APPLE)
    link_directories(/Library/gurobi1000/macos_universal2/lib)
    include_directories(/Library/gurobi1000/macos_universal2/include)
    link_directories(/usr/local/lib)
    find_package(LLVM REQUIRED CONFIG)
    set(CMAKE_C_COMPILER "${LLVM_TOOLS_BINARY_DIR}/clang")
    set(CMAKE_CXX_COMPILER "${LLVM_TOOLS_BINARY_DIR}/clang++")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")

    set(OpenMP_C "${CMAKE_C_COMPILER}" CACHE STRING "" FORCE)
    set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
    set(OpenMP_C_LIB_NAMES "libomp" CACHE STRING "" FORCE)
    set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)

    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
    set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
    set(OpenMP_CXX_LIB_NAMES "libomp" CACHE STRING "" FORCE)
    set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)

elseif(UNIX)
    link_directories(/opt/gurobi1001/linux64/lib)
    include_directories(/opt/gurobi1001/linux64/include)
    find_package(OpenMP REQUIRED)
endif()

add_library(${PROJECT_NAME} SHARED api.cpp skeletonize.hpp merge.hpp util.hpp)

if (APPLE)
    target_link_libraries(${PROJECT_NAME} 
        Open3D::Open3D NLopt::nlopt xtensor
        CGAL::CGAL CGAL::CGAL_Core
        libgurobi_c++.a libgurobi100.dylib
        libomp.dylib)
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} 
        Open3D::Open3D NLopt::nlopt xtensor
        CGAL::CGAL CGAL::CGAL_Core
        gurobi_c++ gurobi100
        OpenMP::OpenMP_CXX)
endif()
