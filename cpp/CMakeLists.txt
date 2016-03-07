# @author : John Allard
# @date   : Feb 20th 2016
# @info   : top-level cmake file for mlstax. Builds the mlstax shared library.
cmake_minimum_required(VERSION 2.8.9)
set (CMAKE_CXX_STANDARD 14)
project(mlstax)
set(CMAKE_BUILD_TYPE Release)

SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
IF( NOT EIGEN3_INCLUDE_DIR )
    SET( EIGEN3_INCLUDE_DIR "/usr/include/eigen3/" )
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )
include_directories( ${MY_SOURCE_DIR}/src )

file(GLOB SOURCES "src/*.cpp" "src/*.h" )

#Generate the shared library from the sources
add_library(mlstax SHARED ${SOURCES})

install(TARGETS mlstax DESTINATION /usr/lib)

add_executable(tests/unit_tests tests/unit_tests.cpp)

target_link_libraries(tests/unit_tests mlstax)
