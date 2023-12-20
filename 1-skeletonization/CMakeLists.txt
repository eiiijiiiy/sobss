cmake_minimum_required(VERSION 3.12.0)
set (CMAKE_CXX_STANDARD 17)

project(BST)

find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/cmake)
find_package(NLopt HINTS ${CMAKE_INSTALL_PREFIX}/lib/cmake)
find_package(RapidJSON)
find_package(xtensor REQUIRED)

if (UNIX)
	find_package(OpenMP)
endif()

# vote along one primary direction
aux_source_directory(./src SRC_LIST)
add_executable(vote vote.cpp ${SRC_LIST})
target_include_directories(vote PUBLIC ${xtensor_INCLUDE_DIR} ${RAPIDJSON_INCLUDE_DIR})

if (APPLE)
	target_link_libraries(vote 
		Open3D::Open3D 
		NLopt::nlopt 
		xtensor)
elseif(UNIX)
	target_link_libraries(vote 
        Open3D::Open3D 
        NLopt::nlopt 
        xtensor
		OpenMP::OpenMP_CXX)
endif()





