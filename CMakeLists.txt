cmake_minimum_required (VERSION 3.16)
project(Marabou)

set(MARABOU_VERSION 2.0.0)
add_definitions("-DMARABOU_VERSION=\"${MARABOU_VERSION}\"")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##################
## User options ##
##################

option(BUILD_STATIC_MARABOU "Build static Marabou binary" OFF)
option(BUILD_PYTHON "Build Python" ON)
option(FORCE_PYTHON_BUILD "Build Python even if there is only 32-bit Python" OFF)
option(RUN_UNIT_TEST "Run unit tests on build" ON)
option(RUN_REGRESS_TEST "Run regression tests on build" OFF)
option(RUN_SYSTEM_TEST "Run system tests on build" OFF)
option(RUN_MEMORY_TEST "Run cxxtest testing with ASAN ON" OFF)
option(RUN_PYTHON_TEST "Run Python API tests if building with Python" OFF)
option(ENABLE_GUROBI "Enable use the Gurobi optimizer" OFF)
option(ENABLE_OPENBLAS "Do symbolic bound tighting using blas" ON) # Not available on Windows
option(CODE_COVERAGE "Add code coverage" OFF)  # Available only in debug mode

###################
## Git variables ##
###################
# Makes the current branch and commit hash accessible in C++ code.

# Get the name of the working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions("-DGIT_BRANCH=\"${GIT_BRANCH}\"")

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions("-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

############################
## Conda env detection    ##
############################

if(DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED ENV{CONDA_BUILD})
  list(INSERT CMAKE_PREFIX_PATH 0 "$ENV{CONDA_PREFIX}")
  message(STATUS "Conda environment detected: $ENV{CONDA_PREFIX}")
elseif(DEFINED ENV{CONDA_BUILD})
  list(INSERT CMAKE_PREFIX_PATH 0 "$ENV{PREFIX}" "$ENV{BUILD_PREFIX}")
  message(STATUS "Conda build environment detected")
endif()

#########################
## Basic configuration ##
#########################

set(DEPS_DIR "${PROJECT_SOURCE_DIR}/deps")
set(TOOLS_DIR "${PROJECT_SOURCE_DIR}/tools")
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(RESOURCES_DIR "${PROJECT_SOURCE_DIR}/resources")
set(REGRESS_DIR "${PROJECT_SOURCE_DIR}/regress")
set(ENGINE_DIR "${SRC_DIR}/engine")
set(COMMON_DIR "${SRC_DIR}/common")
set(BASIS_DIR "${SRC_DIR}/basis_factorization")

if (MSVC)
  set(SCRIPT_EXTENSION bat)
else()
  set(SCRIPT_EXTENSION sh)
endif()

##########
## CVC4 ##
##########

set(CVC4_DIR "${DEPS_DIR}/CVC4")
set(CONTEXT_DIR "${CVC4_DIR}/context")
set(CVC4_BASE_DIR "${CVC4_DIR}/base")
file(GLOB DEPS_CVC4_CONTEXT "${CONTEXT_DIR}/*.cpp")
file(GLOB DEPS_CVC4_BASE "${CVC4_BASE_DIR}/*.cpp")
include_directories(SYSTEM ${CVC4_DIR} ${CVC4_DIR}/include)

###########
## Boost ##
###########

add_definitions(-DBOOST_NO_CXX98_FUNCTION_BASE)
find_package(Boost 1.84.0 REQUIRED COMPONENTS program_options timer chrono thread)
set(LIBS_INCLUDES ${Boost_INCLUDE_DIRS})
list(APPEND LIBS ${Boost_LIBRARIES})

##############
## Protobuf ##
##############

find_package(Protobuf 3.19.2 REQUIRED)
list(APPEND LIBS protobuf::libprotobuf)

##########
## ONNX ##
##########

# NOTE: The ONNX version must be kept in sync with Python `pyproject.toml` and `test_requirements.txt`.
set(ONNX_VERSION 1.15.0)
set(ONNX_DIR "${TOOLS_DIR}/onnx-${ONNX_VERSION}")

if(NOT EXISTS "${ONNX_DIR}/onnx.proto3.pb.h")
    message("generating ONNX protobuf file")
    execute_process(COMMAND ${TOOLS_DIR}/download_onnx.sh ${ONNX_VERSION} ${Protobuf_PROTOC_EXECUTABLE})
endif()
file(GLOB DEPS_ONNX "${ONNX_DIR}/*.cc")
include_directories(SYSTEM ${ONNX_DIR})

############
## Gurobi ##
############

if (${ENABLE_GUROBI})
  message(STATUS "Using Gurobi for LP relaxation for bound tightening")
  if (NOT DEFINED GUROBI_DIR)
    set(GUROBI_DIR $ENV{GUROBI_HOME})
  endif()
  add_compile_definitions(ENABLE_GUROBI)

  set(GUROBI_LIB1 "gurobi_c++")
  set(GUROBI_LIB2 "gurobi110")

  add_library(${GUROBI_LIB1} SHARED IMPORTED)
  set_target_properties(${GUROBI_LIB1} PROPERTIES IMPORTED_LOCATION ${GUROBI_DIR}/lib/libgurobi_c++.a)
  list(APPEND LIBS ${GUROBI_LIB1})
  target_include_directories(${GUROBI_LIB1} INTERFACE ${GUROBI_DIR}/include/)

  add_library(${GUROBI_LIB2} SHARED IMPORTED)

  # MACOSx uses .dylib instead of .so for its Gurobi downloads.
  if (APPLE)
    set_target_properties(${GUROBI_LIB2} PROPERTIES IMPORTED_LOCATION ${GUROBI_DIR}/lib/libgurobi110.dylib)
  else()
    set_target_properties(${GUROBI_LIB2} PROPERTIES IMPORTED_LOCATION ${GUROBI_DIR}/lib/libgurobi110.so)
  endif ()

  list(APPEND LIBS ${GUROBI_LIB2})
  target_include_directories(${GUROBI_LIB2} INTERFACE ${GUROBI_DIR}/include/)
endif()

###########
## cuOpt ##
###########

option(ENABLE_CUOPT "Enable NVIDIA cuOpt GPU LP solver" OFF)

if (${ENABLE_CUOPT})
  message(STATUS "Using cuOpt for GPU LP solving")
  find_package(cuopt REQUIRED)
  add_compile_definitions(ENABLE_CUOPT)
  list(APPEND LIBS cuopt::cuopt)
endif()

##############
## OpenBLAS ##
##############

if (NOT MSVC AND ${ENABLE_OPENBLAS})
  message(STATUS "Using OpenBLAS for matrix multiplication")
  add_compile_definitions(ENABLE_OPENBLAS)
  find_library(OPENBLAS_LIB openblas REQUIRED)
  find_path(OPENBLAS_INCLUDE_DIR openblas_config.h PATH_SUFFIXES openblas)
  if (NOT OPENBLAS_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find OpenBLAS headers (openblas_config.h)")
  endif()
  list(APPEND LIBS ${OPENBLAS_LIB})
  list(APPEND LIBS_INCLUDES ${OPENBLAS_INCLUDE_DIR})
endif()

###########
## Build ##
###########

set(MARABOU_LIB MarabouHelper)

set(BIN_DIR "${CMAKE_BINARY_DIR}/bin")

set(COMMON_REAL "${COMMON_DIR}/real")
set(COMMON_MOCK "${COMMON_DIR}/mock")
file(GLOB SRCS_COMMON_REAL "${COMMON_REAL}/*.cpp")
file(GLOB SRCS_COMMON_MOCK "${COMMON_MOCK}/*.cpp")

set(ENGINE_REAL "${ENGINE_DIR}/real")
set(ENGINE_MOCK "${ENGINE_DIR}/mock")
file(GLOB SRCS_ENGINE_REAL "${ENGINE_REAL}/*.cpp")
file(GLOB SRCS_ENGINE_MOCK "${ENGINE_MOCK}/*.cpp")

set(MPS_PARSER mps)
set(ACAS_PARSER acas)
set(BERKELEY_PARSER berkeley)
set(INPUT_PARSERS_DIR input_parsers)

#-----------------------------------------------------------------------------#
# Determine number of threads available, used to configure (default) parallel
# execution of custom test targets (can be overriden with ARGS=-jN).

include(ProcessorCount)
ProcessorCount(CTEST_NTHREADS)
if(CTEST_NTHREADS EQUAL 0)
  set(CTEST_NTHREADS 1)
endif()

# --------------- set build type ----------------------------
set(BUILD_TYPES Release Debug MinSizeRel RelWithDebInfo)

# Set the default build type to Production
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release CACHE STRING "Options are: Release Debug MinSizeRel RelWithDebInfo" FORCE)
  # Provide drop down menu options in cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${BUILD_TYPES})
endif()
message(STATUS "Building ${CMAKE_BUILD_TYPE} build")

#-------------------------set code coverage----------------------------------#
# Allow coverage only in debug mode only in gcc
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Building with code coverage")
  set(COVERAGE_COMPILER_FLAGS "-g -O0 --coverage" CACHE INTERNAL "")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_COMPILER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# We build a static library that is the core of the project, the link it to the
# API's (executable and python at the moment)
add_library(${MARABOU_LIB} ${DEPS_ONNX} ${DEPS_CVC4_CONTEXT} ${DEPS_CVC4_BASE} ${SRCS_COMMON_REAL} ${SRCS_ENGINE_REAL})
target_include_directories(${MARABOU_LIB} PRIVATE SYSTEM)

set(MARABOU_EXE Marabou${CMAKE_EXECUTABLE_SUFFIX})

add_executable(${MARABOU_EXE} "${ENGINE_DIR}/main.cpp")
set(MARABOU_EXE_PATH "${BIN_DIR}/${MARABOU_EXE}")
add_custom_command(TARGET ${MARABOU_EXE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${MARABOU_EXE}> ${MARABOU_EXE_PATH} )

set(MPS_PARSER_PATH "${BIN_DIR}/${MPS_PARSER}")

if (NOT MSVC)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(COMPILE_FLAGS  -Wall -Wextra -Werror -MMD -Qunused-arguments -Wno-deprecated-declarations -Wno-unused-but-set-variable )
    elseif (CMAKE_BUILD_TYPE MATCHES "Release")
        set(COMPILE_FLAGS  -Wall )
    else()
        set(COMPILE_FLAGS  -Wall -Wextra -Werror -MMD ) #-Wno-deprecated
    endif()
    set(RELEASE_FLAGS ${COMPILE_FLAGS} -O3) #-Wno-deprecated
endif()

if (RUN_MEMORY_TEST)
    if(NOT MSVC)
        set(MEMORY_FLAGS -fsanitize=address -fno-omit-frame-pointer -O1)
    endif()
endif()

add_definitions(-DRESOURCES_DIR="${RESOURCES_DIR}")

if (NOT MSVC)
    set(DEBUG_FLAGS ${COMPILE_FLAGS} ${MEMORY_FLAGS} -g)
    set(CXXTEST_FLAGS ${DEBUG_FLAGS}  -Wno-ignored-qualifiers)
else()
    set(DEBUG_FLAGS ${COMPILE_FLAGS} ${MEMORY_FLAGS})
    add_definitions(-DNOMINMAX) # remove min max macros
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CXXTEST_FLAGS ${CXXTEST_FLAGS} -Wno-terminate)
endif()

# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
list(APPEND LIBS Threads::Threads)

if (BUILD_STATIC_MARABOU)
  # build a static library
  target_link_libraries(${MARABOU_LIB} ${LIBS} -static)
else()
  target_link_libraries(${MARABOU_LIB} ${LIBS})
endif()

target_include_directories(${MARABOU_LIB} PRIVATE ${LIBS_INCLUDES})
target_compile_options(${MARABOU_LIB} PRIVATE ${RELEASE_FLAGS})

# Build marabou executable
target_link_libraries(${MARABOU_EXE} ${MARABOU_LIB})
target_include_directories(${MARABOU_EXE} PRIVATE ${LIBS_INCLUDES})

######################
## Build Python API ##
######################

set(DEFAULT_PYTHON_VERSION "3" CACHE STRING "Default Python version 2/3")
set(PYTHON_VERSIONS_SUPPORTED 2 3)
list(FIND PYTHON_VERSIONS_SUPPORTED ${DEFAULT_PYTHON_VERSION} index)
if(index EQUAL -1)
    message(FATAL_ERROR "Python version must be one of ${PYTHON_VERSIONS_SUPPORTED}")
endif()

set(PYTHON_API_DIR "${PROJECT_SOURCE_DIR}/maraboupy")
if (NOT PYTHON_LIBRARY_OUTPUT_DIRECTORY)
    set(PYTHON_LIBRARY_OUTPUT_DIRECTORY "${PYTHON_API_DIR}")
endif()

# Determine if we should build Python
set(PYTHON32 FALSE)
if(${BUILD_PYTHON})
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
        "import struct; print(struct.calcsize('@P'));"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE PYTHON_SIZEOF_VOID_P
    ERROR_VARIABLE _PYTHON_ERROR_VALUE)
    # message("PYTHON SIZEOF VOID p ${PYTHON_SIZEOF_VOID_P}")
    if (PYTHON_SIZEOF_VOID_P EQUAL 4 AND NOT ${FORCE_PYTHON_BUILD})
        set(PYTHON32 TRUE)
        message(WARNING "Python version is 32-bit, please use build_python.sh in
        maraboupy folder")
    endif()
endif()
if (${FORCE_PYTHON_BUILD})
    set(BUILD_PYTHON ON)
else()
    if (${BUILD_PYTHON} AND NOT ${PYTHON32})
        set(BUILD_PYTHON ON)
    else()
        set(BUILD_PYTHON OFF)
    endif()
endif()

# Actually build Python
if (${BUILD_PYTHON})
    # This is suppose to set the PYTHON_EXECUTABLE variable
    # First try to find the default python version
    find_package(PythonInterp ${DEFAULT_PYTHON_VERSION})
    if (NOT EXISTS ${PYTHON_EXECUTABLE})
        # If the default didn't work just find any python version
        find_package(PythonInterp REQUIRED)
    endif()

    find_package(pybind11 REQUIRED)

    set(MARABOU_PY MarabouCore)
    pybind11_add_module(${MARABOU_PY} ${PYTHON_API_DIR}/MarabouCore.cpp)

    target_link_libraries(${MARABOU_PY} PRIVATE ${MARABOU_LIB})
    target_include_directories(${MARABOU_PY} PRIVATE ${LIBS_INCLUDES})

    set_target_properties(${MARABOU_PY} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PYTHON_LIBRARY_OUTPUT_DIRECTORY})
    if(NOT MSVC)
        target_compile_options(${MARABOU_LIB} PRIVATE -fPIC ${RELEASE_FLAGS})
    endif()
endif()

#################
## Build tests ##
#################

set(MARABOU_TEST_LIB MarabouHelperTest)

add_library(${MARABOU_TEST_LIB})
set (TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/tests")
file(MAKE_DIRECTORY ${TEST_DIR})

set(CMAKE_PREFIX_PATH "${TOOLS_DIR}/cxxtest")
set(CXXTEST_USE_PYTHON FALSE)
find_package(CxxTest)
if(CXXTEST_FOUND)
    include_directories(${CXXTEST_INCLUDE_DIR})
    enable_testing()
endif()

target_link_libraries(${MARABOU_TEST_LIB} ${MARABOU_LIB} ${LIBS})
target_include_directories(${MARABOU_TEST_LIB} PRIVATE ${LIBS_INCLUDES} )
target_compile_options(${MARABOU_TEST_LIB} PRIVATE ${CXXTEST_FLAGS})

add_custom_target(build-tests ALL)

add_custom_target(check
      COMMAND ctest --output-on-failure -j${CTEST_NTHREADS} $$ARGS
      DEPENDS build-tests build_input_parsers ${MARABOU_EXE})

# Decide which tests to run and execute
set(TESTS_TO_RUN "")
# ctest uses regex, so create the string to look: (unit|system) ...
macro(append_tests_to_run new_val)
    if ("${TESTS_TO_RUN}" STREQUAL "")
        set(TESTS_TO_RUN ${new_val})
    else()
        set(TESTS_TO_RUN "${TESTS_TO_RUN}|${new_val}")
    endif()
endmacro()

if (${RUN_UNIT_TEST})
    append_tests_to_run("unit")
endif()
if (${RUN_REGRESS_TEST})
    append_tests_to_run("regress[0-5]")
endif()
if (${RUN_SYSTEM_TEST})
    append_tests_to_run("system")
endif()
if (NOT ${TESTS_TO_RUN} STREQUAL "")
    # make ctest verbose
    set(CTEST_OUTPUT_ON_FAILURE 1)
    add_custom_command(
        TARGET build-tests
        POST_BUILD
        COMMAND ctest --output-on-failure  -L "\"(${TESTS_TO_RUN})\"" -j${CTEST_NTHREADS} $$ARGS
    )
endif()

if (${BUILD_PYTHON} AND ${RUN_PYTHON_TEST})
    if (MSVC)
        add_custom_command(
            TARGET build-tests
            POST_BUILD
            COMMAND cp ${PYTHON_API_DIR}/Release/* ${PYTHON_API_DIR}
        )
    endif()

    add_custom_command(
        TARGET build-tests
        POST_BUILD
        COMMAND ${PYTHON_EXECUTABLE} -m pytest ${PYTHON_API_DIR}/test
    )
endif()

# Add the input parsers
add_custom_target(build_input_parsers)
add_dependencies(build_input_parsers ${MPS_PARSER} ${ACAS_PARSER}
    ${BERKELEY_PARSER})

add_subdirectory(${TOOLS_DIR})
add_subdirectory(${SRC_DIR})
add_subdirectory(${REGRESS_DIR})
