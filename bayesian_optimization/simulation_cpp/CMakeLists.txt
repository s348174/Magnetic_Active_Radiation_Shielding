cmake_minimum_required(VERSION 3.14)
project(simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# --- Python ---
set(Python3_ROOT_DIR $ENV{VIRTUAL_ENV})
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# --- pybind11 ---
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "pybind11 dir: ${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# --- Threads ---
find_package(Threads REQUIRED)
#list(APPEND ${CMAKE_PROJECT_NAME}_LINKED_LIBRARIES PRIVATE Threads::Threads)

# --- Eigen ---
# Option 1: system Eigen
find_package(Eigen3 REQUIRED)

# Option 2: custom Eigen build (Qt style)
# Set this to your Eigen install path
# set(EIGEN3_INCLUDE_DIR "/path/to/eigen")

#include_directories(
#    #${EIGEN3_INCLUDE_DIR}
#    ${PROJECT_SOURCE_DIR}/src
#)

# --- Sources ---
add_subdirectory(src)
list(APPEND ${CMAKE_PROJECT_NAME}_SOURCES ${${CMAKE_PROJECT_NAME}_sources})
list(APPEND ${CMAKE_PROJECT_NAME}_HEADERS ${${CMAKE_PROJECT_NAME}_headers})
list(APPEND ${CMAKE_PROJECT_NAME}_INCLUDE ${${CMAKE_PROJECT_NAME}_includes})

# --- Python module ---
pybind11_add_module(simulator
    bindings/pybind_module.cpp
    ${simulator_sources}
)

target_include_directories(simulator
    PRIVATE
    src  # this folder contains all .hpp files
)

target_link_libraries(simulator
    PRIVATE
    Eigen3::Eigen
    Threads::Threads
    pybind11::pybind11
    Python3::Python
)

# Force .so output to Release folder
#set_target_properties(simulator PROPERTIES
#    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release
#)

# --- Optional standalone executable ---
# --- Executable ---
add_executable(main_exec
    main.cpp
    ${${CMAKE_PROJECT_NAME}_SOURCES}
    ${${CMAKE_PROJECT_NAME}_HEADERS}
)

target_link_libraries(main_exec
    PRIVATE
    Eigen3::Eigen
    Threads::Threads
    pybind11::pybind11
    Python3::Python
)

target_include_directories(main_exec PRIVATE src)

include(CTest)
if(BUILD_TESTING)
    add_executable(relativistic_tests
        tests/relativistic_tests.cpp
        src/Utils.cpp
        src/Sobol.cpp
    )
    target_include_directories(relativistic_tests PRIVATE src)
    target_link_libraries(relativistic_tests PRIVATE Eigen3::Eigen Threads::Threads)
    add_test(NAME relativistic_numerics COMMAND relativistic_tests)
    add_test(NAME stratified_sampling
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=$<TARGET_FILE_DIR:simulator>:${CMAKE_CURRENT_SOURCE_DIR}/../optimizer_py"
            "PYTHONDONTWRITEBYTECODE=1"
            "MPLBACKEND=Agg"
            "MPLCONFIGDIR=${CMAKE_CURRENT_BINARY_DIR}/matplotlib-cache"
            "XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/cache"
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../optimizer_py/tests/test_stratified_sampling.py
    )
    set_tests_properties(stratified_sampling PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../optimizer_py")
    add_test(NAME physical_input_maps
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=$<TARGET_FILE_DIR:simulator>:${CMAKE_CURRENT_SOURCE_DIR}/../optimizer_py"
            "PYTHONDONTWRITEBYTECODE=1"
            "MPLBACKEND=Agg"
            "MPLCONFIGDIR=${CMAKE_CURRENT_BINARY_DIR}/matplotlib-cache"
            "XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/cache"
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../optimizer_py/tests/test_input_maps.py
    )
    set_tests_properties(physical_input_maps PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../optimizer_py")
    add_test(NAME multi_turbo_validation
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=$<TARGET_FILE_DIR:simulator>:${CMAKE_CURRENT_SOURCE_DIR}/../optimizer_py"
            "PYTHONDONTWRITEBYTECODE=1"
            "MPLBACKEND=Agg"
            "MPLCONFIGDIR=${CMAKE_CURRENT_BINARY_DIR}/matplotlib-cache"
            "XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/cache"
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../optimizer_py/tests/test_multi_turbo.py
    )
    set_tests_properties(multi_turbo_validation PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../optimizer_py")
endif()
