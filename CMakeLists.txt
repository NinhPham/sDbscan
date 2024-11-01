cmake_minimum_required(VERSION 3.27)
project(sDbscan)

set(CMAKE_CXX_STANDARD 17)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.71 REQUIRED NO_MODULE)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(sDbscan
        src/main.cpp
        src/Utilities.cpp
        src/sDbscan.cpp
        src/fast_copy.c
        src/fht.c
        )

# Set optimization flags
if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
elseif (MSVC)
    # For Visual Studio
    # You might need /Ox instead of /O3, depending on your Visual Studio version
    # /Ox generally enables maximum optimization
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox /arch:AVX2")
endif()

target_link_libraries (sDbscan Eigen3::Eigen)