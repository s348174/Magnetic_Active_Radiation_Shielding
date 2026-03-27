cmake_minimum_required(VERSION 3.14)
project(Simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Python (from venv) ---
set(Python3_ROOT_DIR $ENV{VIRTUAL_ENV})
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# --- pybind11 ---
find_package(pybind11 REQUIRED)

# --- Threads ---
find_package(Threads REQUIRED)
list(APPEND ${CMAKE_PROJECT_NAME}_LINKED_LIBRARIES PRIVATE Threads::Threads)

# --- Eigen ---
# Option 1: system Eigen
find_package(Eigen3 REQUIRED)

# Option 2: custom Eigen build (Qt style)
# Set this to your Eigen install path
# set(EIGEN3_INCLUDE_DIR "/path/to/eigen")

include_directories(
    #${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

# --- Sources ---
list(APPEND ${CMAKE_PROJECT_NAME}_SOURCES ${${CMAKE_PROJECT_NAME}_sources})
list(APPEND ${CMAKE_PROJECT_NAME}_HEADERS ${${CMAKE_PROJECT_NAME}_headers})
list(APPEND ${CMAKE_PROJECT_NAME}_INCLUDE ${${CMAKE_PROJECT_NAME}_includes})

# --- Python module ---
pybind11_add_module(simulator
    bindings/pybind_module.cpp
    ${SOURCES}
)

# --- Optional standalone executable ---
add_executable(main_exec main.cpp ${${CMAKE_PROJECT_NAME}_SOURCES} ${${CMAKE_PROJECT_NAME}_HEADERS})
target_include_directories(main_exec PRIVATE src)
