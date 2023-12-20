cmake_minimum_required(VERSION 3.12.0)
set (CMAKE_CXX_STANDARD 17)

project(merging)

find_package(Qt5 COMPONENTS Core Gui OpenGL Widgets REQUIRED)
find_package(CGAL COMPONENTS Core Qt5 REQUIRED)
find_package(Eigen3 REQUIRED)

if (APPLE)
    link_directories(/Library/gurobi1000/macos_universal2/lib)
    include_directories(/Library/gurobi1000/macos_universal2/include)
elseif(UNIX)
    link_directories(/opt/gurobi1001/linux64/lib)
    include_directories(/opt/gurobi1001/linux64/include)
    find_package(OpenMP)
endif()

add_executable(merge merge.cpp)

if (APPLE)
    target_link_libraries(merge 
        Qt5::Widgets Qt5::OpenGL Qt5::Core Qt5::Gui
        CGAL::CGAL CGAL::CGAL_Core CGAL::CGAL_Qt5
        libgurobi_c++.a libgurobi100.dylib)
elseif(UNIX)
    target_link_libraries(merge 
        Qt5::Widgets Qt5::OpenGL Qt5::Core Qt5::Gui
        CGAL::CGAL CGAL::CGAL_Core CGAL::CGAL_Qt5
        libgurobi_c++.a libgurobi100.so
        OpenMP::OpenMP_CXX)
endif()




