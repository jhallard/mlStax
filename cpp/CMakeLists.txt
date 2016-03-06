# @author : John Allard
# @date   : Feb 20th 2016
# @info   : top-level cmake file for mlstax. Builds the mlstax shared library.
cmake_minimum_required(VERSION 2.8.9)
set (CMAKE_CXX_STANDARD 11)
project(mlstax)
set(CMAKE_BUILD_TYPE Release)

SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
IF( NOT EIGEN3_INCLUDE_DIR )
        MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
    ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )
include_directories(include)

file(GLOB SOURCES "src/*.cpp")

#Generate the shared library from the sources
add_library(mlstax SHARED ${SOURCES})

install(TARGETS mlstax DESTINATION /usr/lib)
