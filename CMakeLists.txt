cmake_minimum_required(VERSION 3.17)
project(Place_recognization)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/homebrew/Cellar/qt@5/5.15.2/lib/cmake/Qt5")

find_package(OpenCV 3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(vtk 9 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${VTK_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(Place_recognization main.cpp projection.cpp projection.h spatial_calculation.h spatial_calculation.cpp extract_segments.h extract_segments.cpp)
target_link_libraries(Place_recognization ${OpenCV_LIBS} ${Boost_LIBS} ${PCL_LIBS} ${VTK_LIBS})