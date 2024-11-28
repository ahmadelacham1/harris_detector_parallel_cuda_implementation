cmake_minimum_required(VERSION 3.12)

# set the project name and version
project(HarrisCornerDetector C CXX CUDA)

# set CXX standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Linear algebra package
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

# CUDA Libraries
find_package(CUDA)
if(CUDA_FOUND)
  enable_language(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS} )
endif()
add_executable(sequential ./src/sequential.cpp)
target_link_libraries(sequential hcd hcd_cuda png tiff jpeg m)
add_executable(parallel ./src/parallel.cpp)
target_link_libraries(parallel hcd hcd_cuda png tiff jpeg m)
