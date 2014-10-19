CMAKE_MINIMUM_REQUIRED(VERSION 2.8.4)
PROJECT(Test)

SET(CMAKE_MODULE_PATH "/usr/share/cmake-2.8/Modules/")
SET(CMAKE_PREFIX_PATH "~/local")

FIND_PACKAGE(Eigen3 REQUIRED)
include_directories("${EIGEN3_INCLUDE_DIR}")
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

ADD_LIBRARY(MyLib include/saveMesh.cpp include/ply.cpp)

ADD_EXECUTABLE(MyRansac MyRansac.cpp)
ADD_EXECUTABLE(MyAContrarioRansac MyAContrarioRansac.cpp)
ADD_EXECUTABLE(BestTextureFilter BestTextureFilter.cpp)
ADD_EXECUTABLE(MultiBandPanoramaBlending MultiBandPanoramaBlending.cpp)
ADD_EXECUTABLE(PanoramaBlending PanoramaBlending.cpp)
ADD_EXECUTABLE(PanoramaBlending2 PanoramaBlending2.cpp)
ADD_EXECUTABLE(CeresTest CeresTest.cpp)
ADD_EXECUTABLE(CeresJacobianTest CeresJacobianTest.cpp)
ADD_EXECUTABLE(CeresBundleAdjuster simple_bundle_adjuster.cc)
ADD_EXECUTABLE(ManualJacobianBundleAdjuster ManualJacobianBundleAdjuster.cpp)
ADD_EXECUTABLE(CeresDTM CeresDTM.cpp)

TARGET_LINK_LIBRARIES(MyRansac ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(MyAContrarioRansac ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(BestTextureFilter ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(MultiBandPanoramaBlending ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(PanoramaBlending ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(PanoramaBlending2 ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(CeresTest ${OpenCV_LIBS} ${CERES_LIBRARIES})
TARGET_LINK_LIBRARIES(CeresJacobianTest ${OpenCV_LIBS} ${CERES_LIBRARIES})
TARGET_LINK_LIBRARIES(CeresBundleAdjuster ${CERES_LIBRARIES} MyLib)
TARGET_LINK_LIBRARIES(ManualJacobianBundleAdjuster ${CERES_LIBRARIES} MyLib)
TARGET_LINK_LIBRARIES(CeresDTM ${CERES_LIBRARIES} MyLib)