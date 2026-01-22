cmake_minimum_required(VERSION 4.0)
project(fractals)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG         0fa201c969e48ecc253581c5841ce73f44d42f49
)
FetchContent_MakeAvailable(SFML)

add_executable(fractals main.cpp)
target_link_libraries(fractals PUBLIC SFML::Graphics SFML::Audio)