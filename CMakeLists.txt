cmake_minimum_required(VERSION 3.30)
project(FilterProject)

set(CMAKE_CXX_STANDARD 20)

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FilterProject
        main.cpp
        src/effects.cpp
        src/effects.h
        src/test.cpp
        src/test.h
)

target_link_libraries(FilterProject ${OpenCV_LIBS})

