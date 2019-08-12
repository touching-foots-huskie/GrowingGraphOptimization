cmake_minimum_required(VERSION 3.0)
project(GraphOptimization CXX C)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

### install lib, only do that if *.c or *.cpp exist
# since C srcs exist, use *.c* as pattern, not *.cpp

# Eigen structure
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Find ceres component
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

# export cmake setting to the source code
include_directories(${PROJECT_SOURCE_DIR}/include)

# library
add_library(graph_optimizer SHARED ${PROJECT_SOURCE_DIR}/src/graph_optimizer.cpp)
add_library(graph_constraints SHARED ${PROJECT_SOURCE_DIR}/src/graph_constraints.cpp)

# linking with dependences
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES} Eigen3::Eigen graph_optimizer graph_constraints)
