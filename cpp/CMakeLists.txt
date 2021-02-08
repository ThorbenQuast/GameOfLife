# Setup the project
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(GAMEOFLIFE)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

find_package(ROOT REQUIRED)

include(${ROOT_USE_FILE})

include_directories(${PROJECT_SOURCE_DIR}/include ${ROOT_INCLUDE_DIRS})


#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)


add_executable(gol main.cpp ${sources} ${headers})
target_link_libraries(gol ${ROOT_LIBRARIES})