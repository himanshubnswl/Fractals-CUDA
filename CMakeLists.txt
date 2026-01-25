cmake_minimum_required(VERSION 4.0)
project(fractals LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)


if(WIN32 OR WIN64)
    if(NOT ${CMAKE_CXX_COMPILER_ID} EQUAL MSVC)
        message(compatible MSVC needed for windows!)
    endif()
endif()

#find_package(CUDAToolKit REQUIRED)
find_package(CUDAToolkit REQUIRED)
include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG         0fa201c969e48ecc253581c5841ce73f44d42f49
)
FetchContent_MakeAvailable(SFML)

add_executable(fractals main.cpp mandelbrot.cu)
target_link_libraries(fractals PUBLIC SFML::Graphics SFML::Audio CUDA::cudart)