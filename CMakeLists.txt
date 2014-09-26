cmake_minimum_required(VERSION 2.8.4)
project(Test)

find_package(OpenCV REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(MyRansac MyRansac.cpp)
add_executable(MyAContrarioRansac MyAContrarioRansac.cpp)

target_link_libraries(MyRansac ${OpenCV_LIBS})
target_link_libraries(MyAContrarioRansac ${OpenCV_LIBS})