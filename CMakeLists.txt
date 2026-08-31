######################################################################
# Set version and policies 
######################################################################
cmake_minimum_required(VERSION 3.18.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Necessary for cmake try_compile of Fortran/C Interface
# when mixing gfortran and clang
# CNY: Any workaround?
set(CMAKE_Fortran_FLAGS_INIT "-fPIC")

#######################
# Check CoQui git info
#######################
execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT GIT_COMMIT_HASH)
  set(GIT_COMMIT_HASH "unknown")
endif()

execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH_NAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT GIT_BRANCH_NAME)
  set(GIT_BRANCH_NAME "unknown")
endif()
set(GIT_BRANCH_NAME "${GIT_BRANCH_NAME}" CACHE STRING "Git branch name of the source tree")
set(GIT_COMMIT_HASH "${GIT_COMMIT_HASH}" CACHE STRING "Git commit hash of CoQuí")
message(STATUS "CoQuí @ git branch ${GIT_BRANCH_NAME}, commit ${GIT_COMMIT_HASH}")

######################################################################
# Set project and version 
######################################################################

set(COQUI_VERSION_MAJOR 0)
set(COQUI_VERSION_MINOR 3)
set(COQUI_VERSION_PATCH 0)
set(COQUI_VERSION "${COQUI_VERSION_MAJOR}.${COQUI_VERSION_MINOR}.${COQUI_VERSION_PATCH}")

project(COQUI VERSION ${COQUI_VERSION} LANGUAGES C CXX Fortran)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(PROJECT_CMAKE ${CMAKE_SOURCE_DIR}/cmake)

set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH "Library directory.")
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Binary directory.")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories( ${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src )

######################################################################
# CUDA/HIP build options. Read first, to be able to set policies! 
######################################################################
option(ENABLE_CUDA " Build device code with CUDA backend." OFF)
option(ENABLE_HIP " Build device code with HIP backend." OFF)

######################################################################
# CMake Policies 
######################################################################
# find_package uses <PackageName>_ROOT variables
cmake_policy(SET CMP0074 NEW)
# Include file check macros honor CMAKE_REQUIRED_LIBRARIES. 
cmake_policy(SET CMP0075 NEW)
# Initialize CMAKE_CUDA_ARCHITECTURES when CMAKE_CUDA_COMPILER_ID is NVIDIA. 
if(ENABLE_CUDA OR ENABLE_HIP)
 cmake_policy(SET CMP0104 NEW)
endif()
include(CMakeDependentOption)


######################################################################
# CTest
######################################################################
SET(PROJECT_UNIT_TEST_DIR ${PROJECT_BINARY_DIR}/tests/bin)
enable_testing()
include( CTest )

######################################################################
# Build level
######################################################################
if ( NOT CMAKE_BUILD_TYPE ) 
  set( CMAKE_BUILD_TYPE Release )
endif()
string( TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
message( STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

######################################################################
# Check C++ standard and stdlib. Currently requires 20. 
######################################################################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

######################################################################
# set options
######################################################################
option(ENABLE_ROCM " Build device code with ROCM backend" OFF)
option(ENABLE_OPENMP " Build device code with OPENMP backend" OFF)
option(ENABLE_FFTW "Build using FFTW3 library." ON)
option(ENABLE_HEFFTE "Build using heFFTe library." OFF)
option(ENABLE_SLATE "Build using Slate library." ON)
option(ENABLE_CUTENSOR "Build using cutensor library. Must set CUTENSOR_ROOT" OFF)
option(ENABLE_TBLIS "Build using tblis library. Must set TBLIS_ROOT" OFF)
option(ENABLE_SPDLOG "Build with spdlog" ON)
option(ENABLE_SPHERICART "Build with shericart" OFF)
option(ENABLE_WANNIER90 "Build with wannier90 support." ON)
option(ENABLE_DLR "Build with cppdlr backend for IAFT." ON)
option(REQUIRE_NCCL "Require NCCL library with CUDA backend." OFF)
option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_STATIC "Link to static libraries" OFF)
option(ENABLE_TIMERS "Enable internal timers" ON)
option(COMPILE_NDA_TESTS "Build with nda tests" OFF)
option(NDA_PYTHON_SUPPORT "Build nda with python support" OFF)
option(ENABLE_CPPTRACE "Build with cpptrace support" ON)
option(FETCH_CCCL "Fetch CCCL and enable support. Only used with ENABLE_CUDA" OFF)
option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." ON)
option(COQUI_PYTHON_SUPPORT "Build with Python interface" OFF)
option(COQUI_UPDATE_PYTHON_BINDINGS "Generate coqui python bindings from scratch" OFF)

if(COQUI_PYTHON_SUPPORT)
  # Set PIC for all libraries
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

#option(USE_DEVICE_COMPILER "Use gpu compiler, e.g. nvcc, for all files." OFF)
set(USE_DEVICE_COMPILER OFF)

if( ENABLE_CUDA OR ENABLE_HIP OR ENABLE_OPENMP)
  set(ENABLE_DEVICE ON)
  add_compile_definitions(ENABLE_DEVICE)
else()
  set(ENABLE_DEVICE OFF)
endif() 
if(ENABLE_OPENMP)
  add_compile_definitions(ENABLE_OPENMP)
endif()

######################################################################
# CUDA
######################################################################
set(ENABLE_NCCL FALSE)
if(ENABLE_CUDA)
  enable_language(CUDA)
  if(DEFINED CUDA_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
  else()
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  find_package(CUDAToolkit 12.0.0 REQUIRED)
  if(NOT FETCH_CCCL)
    find_package(CCCL REQUIRED)
  endif()
  add_compile_definitions(ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  string(APPEND CMAKE_CUDA_FLAGS "-g -std=c++20 --expt-relaxed-constexpr --extended-lambda -Xcudafe \"--diag_suppress=implicit_return_from_non_void_function\" -Wdeprecated-declarations -w") 
  ################################
  # CUTENSOR: linked through nda # 
  ################################
  ########
  # NCCL # 
  ########
  # look for nccl by default
  if(NOT DEFINED NCCL_ROOT AND DEFINED ENV{NCCL_ROOT})
    set(NCCL_ROOT $ENV{NCCL_ROOT})
  endif()
  find_package(NCCL QUIET)
  if(${NCCL_FOUND})
    set(ENABLE_NCCL TRUE)
    add_compile_definitions(ENABLE_NCCL)
    message(STATUS "Found NCCL library: ${NCCL_LIBRARIES}")
    include_directories(${NCCL_INCLUDE_DIRS})
    link_libraries(${NCCL_LIBRARIES})
  else()
    # try by hand
    find_path(NCCL_INCLUDE_DIRS NAMES nccl.h nccl_net.h HINTS ${NCCL_ROOT}/include)
    find_library(NCCL_LIBRARIES NAMES nccl HINTS ${NCCL_ROOT}/lib)
    if(NCCL_INCLUDE_DIRS AND NCCL_LIBRARIES)
      message(STATUS "Found NCCL library: ${NCCL_LIBRARIES}")
      add_compile_definitions(ENABLE_NCCL)
      include_directories(${NCCL_INCLUDE_DIRS})
      link_libraries(${NCCL_LIBRARIES})
      set(NCCL_FOUND TRUE)
    endif()
  endif()
  if(REQUIRE_NCCL AND NOT NCCL_FOUND)
    message(FATAL_ERROR "NCCL requested but not found. Set NCCL_ROOT")
  endif()
endif()

######################################################################
# ROCM 
######################################################################
if(ENABLE_ROCM)
  message(STATUS "ROCM_ROOT: ${ROCM_ROOT}")
  add_compile_definitions(ENABLE_ROCM)
  add_library(ROCM::libraries INTERFACE IMPORTED)
  # temporarily put rocsolver rocrand here for convenience, should be moved to Platforms.
  set_target_properties(ROCM::libraries PROPERTIES INTERFACE_INCLUDE_DIRECTORIES  "${ROCM_ROOT}/include"
                                                   INTERFACE_LINK_LIBRARIES "-L${ROCM_ROOT}/lib;-lrocsolver;-lrocrand")
endif(ENABLE_ROCM)

######################################################################
#  HIP 
######################################################################
if(ENABLE_HIP)
  if(NOT ENABLE_ROCM)
    message(FATAL_ERROR "ROCM is required to use HIP. Please set ENABLE_ROCM=ON.")
  endif()
  set(CMAKE_MODULE_PATH "${ROCM_ROOT}/hip/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP REQUIRED)

  add_compile_definitions(ENABLE_HIP)
  add_library(HIP::HIP INTERFACE IMPORTED)
  # temporarily put hipsparse hipblas here for convenience, should be moved to Platforms.
  set_target_properties(HIP::HIP PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ROCM_ROOT}/include"
                                            INTERFACE_COMPILE_DEFINITIONS "ENABLE_HIP"
                                            INTERFACE_LINK_LIBRARIES "-L${ROCM_ROOT}/lib;-lhipsparse;-lhipblas;-lamdhip64")
endif(ENABLE_HIP)

######################################################################
# MPI
######################################################################
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
link_libraries(MPI::MPI_CXX) 
message(STATUS "MPI Executable for running programs:" ${MPIEXEC_EXECUTABLE})
message(STATUS "MPI number of processors detected on the host system: " ${MPIEXEC_MAX_NUMPROCS})
message(STATUS "MPI nproc flag: " ${MPIEXEC_NUMPROC_FLAG})
message(STATUS "MPI exec flags: " ${MPIEXEC_PREFLAGS})
if(DEFINED CTEST_NPROC) 
  if( ${CTEST_NPROC}.gt.${MPIEXEC_MAX_NUMPROCS} )
    set(CTEST_NPROC ${MPIEXEC_MAX_NUMPROCS})
  endif()
else()
  set(CTEST_NPROC 1)
endif()
message( STATUS "Number of processors using ctest parallel runs: ${CTEST_NPROC} ") 
include(CMakePrintHelpers)
cmake_print_properties(TARGETS MPI::MPI_CXX PROPERTIES INTERFACE_LINK_LIBRARIES INTERFACE_INCLUDE_DIRECTORIES)

######################################################################
# BLAS/LAPACK 
######################################################################
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
#cmake_print_properties(TARGETS BLAS::BLAS PROPERTIES INTERFACE_LINK_LIBRARIES) 
#cmake_print_properties(TARGETS LAPACK::LAPACK PROPERTIES INTERFACE_LINK_LIBRARIES)

######################################################################
# FFTW: not using MKL's FFTW yet, figure out how if requested 
######################################################################
if(ENABLE_FFTW)
  find_package(FFTW REQUIRED DOUBLE_OPENMP_LIB)
  add_compile_definitions(ENABLE_FFTW)
endif()

######################################################################
# HEFFTE: Highly Efficient FFT for Exascale
######################################################################
if(ENABLE_HEFFTE)
  find_package(Heffte REQUIRED)
  add_compile_definitions(ENABLE_FEFFTE)
endif()

######################################################################
# HDF5
######################################################################
find_package(ZLIB)
set(HDF5_PREFER_PARALLEL TRUE CACHE BOOL "Request parallel/serial HDF5 library")
find_package(HDF5 REQUIRED COMPONENTS C)
message(STATUS "HDF5 version: ${HDF5_VERSION}")
if(HDF5_IS_PARALLEL)
  message(STATUS "Parallel HDF5 library found")
  set(ENABLE_PHDF5 TRUE CACHE BOOL "Enable code paths using parallel HDF5")
else(HDF5_IS_PARALLEL)
  message(STATUS "Serial HDF5 library found")
  set(ENABLE_PHDF5 FALSE CACHE BOOL "Enable code paths using parallel HDF5")
endif(HDF5_IS_PARALLEL)
add_library(IO::HDF5 INTERFACE IMPORTED)
set_target_properties(IO::HDF5 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}"
                                          INTERFACE_LINK_LIBRARIES "${HDF5_LIBRARIES}")
if(ENABLE_PHDF5)
  set_property(TARGET IO::HDF5 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "ENABLE_PHDF5")
endif(ENABLE_PHDF5)

######################################################################
# Boost: Header-only
######################################################################
set(Boost_NO_BOOST_CMAKE on)
find_package(Boost 1.77.0)
if(Boost_FOUND)
  message(STATUS "Boost found by find_package.")
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE=1")
  add_definitions(-DBOOST_LOCALE_HIDE_AUTO_PTR)
endif(Boost_FOUND)

######################################################################
# SLATE: Software for Linear Algebra Targeting Exascale
######################################################################
if(ENABLE_SLATE)
  find_package(slate REQUIRED) 
  add_compile_definitions(ENABLE_SLATE)
  #add_compile_definitions(USE_SLATE_HOSTBATCH ENABLE_SLATE)
else(ENABLE_SLATE)
# empty target ??? 
  add_library(slate INTERFACE) 
endif(ENABLE_SLATE)

######################################################################
#  Now "fetchable" content
######################################################################
include(FetchContent)

######################################################################
# toml++: c++ parser for toml files
######################################################################
if(DEFINED TOMLPP)
  set(FETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS ${TOMLPP})
endif()

FetchContent_Declare(
        tomlplusplus
        GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
        GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

######################################################################
# spdlog: logging library
######################################################################
if(ENABLE_SPDLOG)
  if(DEFINED SPDLOG) 
    set(FETCHCONTENT_SOURCE_DIR_SPDLOG ${SPDLOG})
  endif()

  FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.8.0
  )
  FetchContent_MakeAvailable(spdlog)
  # turn off warnings
  set_property(TARGET spdlog PROPERTY SYSTEM TRUE)
  add_compile_definitions(ENABLE_SPDLOG)
endif()

######################################################################
# nda: a C++ library providing multi-dimensional array classes
######################################################################
if(DEFINED NDA) 
  set(FETCHCONTENT_SOURCE_DIR_NDA ${NDA})
endif()
# set nda options before FetchContent_Declare
set(CutensorSupport ${ENABLE_CUTENSOR} CACHE INTERNAL "")
set(CudaSupport ${ENABLE_CUDA} CACHE INTERNAL "")
set(TblisSupport ${ENABLE_TBLIS} CACHE INTERNAL "")
set(Build_Tests ${COMPILE_NDA_TESTS} CACHE INTERNAL "")
if(COQUI_PYTHON_SUPPORT)
  # nda_py library is needed in CoQui python version
  set(PythonSupport ON CACHE INTERNAL "")
else()
  set(PythonSupport ${NDA_PYTHON_SUPPORT} CACHE INTERNAL "")
endif()
# h5/unstable defaults Update_Python_Bindings=ON, which invokes clair-c2py to
# regenerate its checked-in .wrap.cxx files. CoQui only needs to *compile* h5's
# Python bindings, not regenerate them, so suppress the clair dependency here.
set(Update_Python_Bindings OFF CACHE INTERNAL "")
FetchContent_Declare(
        nda
        GIT_REPOSITORY "https://github.com/triqs/nda.git"
        GIT_TAG        tensor
)
FetchContent_MakeAvailable(nda)

set_property(GLOBAL PROPERTY nda_FOUND TRUE)

if(ENABLE_CUDA AND ENABLE_CUTENSOR)
  add_compile_definitions(ENABLE_CUTENSOR)
endif()
if(ENABLE_TBLIS)
  add_compile_definitions(ENABLE_TBLIS)
endif()

######################################################################
# CCCL from github (useful if CudaToolkit is old 
######################################################################
if(ENABLE_CUDA AND FETCH_CCCL)
  # set nevanlinna options before FetchContent_Declare
  FetchContent_Declare(
        cccl
        GIT_REPOSITORY "https://github.com/NVIDIA/cccl.git"
        GIT_TAG        main 
  )
  FetchContent_MakeAvailable(cccl)
  include_directories(${cccl_SOURCE_DIR}/c++)
endif()

######################################################################
# Alfredo Correa's boost/mpi3 
######################################################################
#if(DEFINED MPI3) 
#  set(FETCHCONTENT_SOURCE_DIR_BMPI3 ${MPI3}) 
#endif()
#FetchContent_Declare(
#        bmpi3 
#        GIT_REPOSITORY "https://gitlab.com/correaa/boost-mpi3.git" 
#        GIT_TAG        master 
#)
#FetchContent_Populate(bmpi3)
#add_subdirectory(${bmpi3_SOURCE_DIR} ${bmpi3_BINARY_DIR} EXCLUDE_FROM_ALL)

######################################################################
# itertools
######################################################################
#FetchContent_Declare(
#        itertools 
#        GIT_REPOSITORY "https://github.com/TRIQS/itertools.git" 
#        GIT_TAG        1.1.x
#)
#FetchContent_MakeAvailable(itertools)

######################################################################
# catch2
######################################################################
if(BUILD_UNIT_TESTS)
  if(DEFINED CATCH2)
    set(FETCHCONTENT_SOURCE_DIR_CATCH2 ${CATCH2})
  endif()

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.10
  )
    #GIT_TAG        v3.4.0
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
  include( Catch )
endif(BUILD_UNIT_TESTS)

######################################################################
# cpptrace 
######################################################################
if(ENABLE_CPPTRACE)
  check_include_file("stdint.h" HAVE_STDINT_H)
  check_include_file("unistd.h" HAVE_UNISTD_H)
  check_include_file("fcntl.h" HAVE_FCNTL_H)
  # MAM: Failing in MacOSX, find correct way
  add_compile_definitions( $<$<COMPILE_LANGUAGE:C>:PACKAGE_VERSION="0"> )
  FetchContent_Declare(
    cpptrace
    GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
    GIT_TAG        v0.8.2 # <HASH or TAG>
  )
  FetchContent_MakeAvailable(cpptrace)
endif(ENABLE_CPPTRACE)

########################
# Find Python, c2py and clair for pybinding
########################
if(COQUI_PYTHON_SUPPORT)
  find_package(Python COMPONENTS Interpreter Development NumPy)
  # Fetch the c2py library
  FetchContent_Declare(
    c2py
    GIT_REPOSITORY https://github.com/flatironinstitute/c2py
    GIT_TAG        unstable
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(c2py)

  if (COQUI_UPDATE_PYTHON_BINDINGS)
    find_program(clair-c2py clair-c2py REQUIRED)
  endif()
endif()

######################################################################
# lab-cosmo sphericart 
######################################################################
if(ENABLE_SPHERICART)
  if(DEFINED SPHERICART)
    set(FETCHCONTENT_SOURCE_DIR_SPHERICART ${SPHERICART})
  endif()

  #set(SPHERICART_OPENMP OFF CACHE INTERNAL "")
  FetchContent_Declare(
      sphericart
      GIT_REPOSITORY https://github.com/lab-cosmo/sphericart.git 
      GIT_TAG        main # <HASH or TAG>
   )
   FetchContent_MakeAvailable(sphericart)
   add_compile_definitions(ENABLE_SPHERICART)
endif(ENABLE_SPHERICART)

######################################################################
# Wannier90 
######################################################################
if(ENABLE_WANNIER90)
  if(DEFINED WANNIER90)
    set(FETCHCONTENT_SOURCE_DIR_WANNIER90 ${WANNIER90})
  endif()

  FetchContent_Declare(
      Wannier90 
      GIT_REPOSITORY https://github.com/wannier-developers/wannier90.git 
      GIT_TAG        3ff086adf9c223c140c54af7070c6e6f5871dbe9 #develop
   )
   FetchContent_MakeAvailable(Wannier90)
    # Add the include directory of the fortran compiler explicitly to ensure ISO_Fortran_binding.h 
    # can be found when building Wannier90_lib. This is a workaround for the issue of missing 
    # ISO_Fortran_binding.h when using clang as the C++ compiler and gfortran as the Fortran compiler, 
    execute_process(
      COMMAND ${CMAKE_Fortran_COMPILER} -print-file-name=include
      OUTPUT_VARIABLE GFORTRAN_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    target_include_directories(Wannier90_lib PRIVATE "${GFORTRAN_INCLUDE_DIR}")
    add_compile_definitions(ENABLE_WANNIER90)
endif(ENABLE_WANNIER90)

######################################################################
# cppdlr: Discrete Lehmann Representation backend (optional)
######################################################################
if(ENABLE_DLR)
  # cppdlr main branch currently checks CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
  # at configure time even as a subproject. Clear it to allow FetchContent use.
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
  endif()

  FetchContent_Declare(
      cppdlr
      GIT_REPOSITORY https://github.com/flatironinstitute/cppdlr.git
      GIT_TAG        f6bd6ab1823403015486c2c6b3399fd4db4468c1 #main
  )
  FetchContent_MakeAvailable(cppdlr)
  # temporary workaround before cppdlr's nda dependency is updated. 
  if(TARGET cppdlr_c)
    target_compile_options(cppdlr_c PRIVATE
      -include ${CMAKE_SOURCE_DIR}/src/numerics/imag_axes_ft/dlr/nda_linalg_compat.hpp
    )
  endif()
  add_compile_definitions(ENABLE_DLR)
endif(ENABLE_DLR)

######################################################################
# Boost
######################################################################
#if(NOT Boost_FOUND)
#  message(STATUS "Boost not found by find_package. Using FetchContent (might take a few minutes). ")
#  if(DEFINED BOOST)
#    set(FETCHCONTENT_SOURCE_DIR_BOOST ${BOOST})
#  endif()
#
#  set(BOOST_INCLUDE_LIBRARIES optional property_tree iterator)
#  FetchContent_Declare(
#    Boost
#    GIT_REPOSITORY https://github.com/boostorg/boost.git
#    GIT_TAG boost-1.80.0
#  )
#  FetchContent_MakeAvailable(Boost)
#  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
#endif()

######################################################################
# Mainly for ninja output 
######################################################################
if(${FORCE_COLORED_OUTPUT})
  add_compile_options(
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<COMPILE_LANGUAGE:CXX>>:-fdiagnostics-color=always>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<COMPILE_LANGUAGE:CXX>>:-fcolor-diagnostics>
  )
endif()

######################################################################
add_compile_options(
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-fconcepts-diagnostics-depth=10>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-narrowing>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>
)

configure_file(${PROJECT_SOURCE_DIR}/src/COQUI.settings ${PROJECT_BINARY_DIR}/bin/COQUI.settings)
install(FILES ${PROJECT_BINARY_DIR}/bin/COQUI.settings DESTINATION bin)

add_subdirectory(src)
