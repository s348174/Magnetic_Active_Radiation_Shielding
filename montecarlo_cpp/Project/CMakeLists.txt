cmake_minimum_required(VERSION 3.18)
project(MARS LANGUAGES CXX C VERSION 1.0.0)

# Set C++ Standard
################################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MARS_ENABLE_CUDA "Enable CUDA acceleration for magnetic field integration" ON)
if (MARS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        message(STATUS "CUDA enabled with compiler: ${CMAKE_CUDA_COMPILER}")
        set(MARS_HAS_CUDA ON)
    else()
        message(WARNING "CUDA compiler not found. Falling back to CPU magnetic field integration.")
        set(MARS_HAS_CUDA OFF)
    endif()
else()
    set(MARS_HAS_CUDA OFF)
endif()

# IMPOSE WARNINGS ON DEBUG
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -pedantic-errors")

# IMPOSE CXX FLAGS FOR WINDOWS
if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
endif (WIN32)

# Add dependencies
################################################################################
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_SOURCE_DIR}/../externals/Main_Install/eigen3)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_SOURCE_DIR}/../externals/Main_Install/googletest)

## Eigen3
find_package(Eigen3 CONFIG REQUIRED)
list(APPEND ${CMAKE_PROJECT_NAME}_LINKED_LIBRARIES PUBLIC Eigen3::Eigen)

## Threads
find_package(Threads REQUIRED)
list(APPEND ${CMAKE_PROJECT_NAME}_LINKED_LIBRARIES PRIVATE Threads::Threads)

# Insert Sources
################################################################################
add_subdirectory(src)

list(APPEND ${CMAKE_PROJECT_NAME}_SOURCES ${${CMAKE_PROJECT_NAME}_sources})
list(APPEND ${CMAKE_PROJECT_NAME}_HEADERS ${${CMAKE_PROJECT_NAME}_headers})
list(APPEND ${CMAKE_PROJECT_NAME}_INCLUDE ${${CMAKE_PROJECT_NAME}_includes})

# Create executable
################################################################################
add_executable(${CMAKE_PROJECT_NAME} main.cpp
    ${${CMAKE_PROJECT_NAME}_SOURCES}
    ${${CMAKE_PROJECT_NAME}_HEADERS}
)

target_link_libraries(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_LINKED_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${${CMAKE_PROJECT_NAME}_INCLUDE})
target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC -fPIC)

if (MARS_HAS_CUDA)
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE MARS_USE_CUDA)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CUDA::cudart)
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/particles_input.csv DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
