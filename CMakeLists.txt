CMAKE_MINIMUM_REQUIRED (VERSION 3.24)

if(WALBERLA_BUILD_WITH_CUDA)
    PROJECT ( TurbineCore LANGUAGES CXX CUDA )
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
        message(FATAL_ERROR "CUDA version must be at least 12.0!  " ${CMAKE_CUDA_COMPILER_VERSION})
    endif()

    find_package(CUDAToolkit REQUIRED)
    set( CMAKE_CUDA_SEPARABLE_COMPILATION ON )

    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    endif()

elseif(WALBERLA_BUILD_WITH_HIP)
    PROJECT ( TurbineCore LANGUAGES CXX HIP )
    enable_language(HIP)
    find_package(hip REQUIRED)
    add_compile_definitions(__HIP_PLATFORM_AMD__)
else()
    PROJECT ( TurbineCore LANGUAGES CXX )
endif()

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -c")

# Extends cmake module path - so that FindwaLBerla.cmake in the current directory is found
set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${TurbineCore_SOURCE_DIR} )
find_package( waLBerla )

enable_testing()
include_directories( src )
include_directories ( ${TurbineCore_BINARY_DIR}/src )

# enable waLBerla-wind Python modules
set( WALBERLA_WIND_PYTHON_DIR ${TurbineCore_SOURCE_DIR}/python )
set( ENV{PYTHONPATH} "${WALBERLA_WIND_PYTHON_DIR}:$ENV{PYTHONPATH}" )

add_subdirectory( src  )
add_subdirectory( apps )

if ( WALBERLA_BUILD_TESTS )
    add_subdirectory ( tests EXCLUDE_FROM_ALL )
endif()
