# oss/CMakeLists.txt - strata-oss, the GPT-OSS engine (HIP).  A separate project: the Qwen engine's build is untouched.
#
#   cmake -S oss -B build-oss -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1201
#   cmake --build build-oss
cmake_minimum_required(VERSION 3.21)
project(strata_oss LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(hip REQUIRED)
find_package(hipblas REQUIRED)

option(STRATA_OSS_NATIVE "Compile the CPU kernels for this machine's CPU (-march=native)" ON)

add_executable(strata-oss oss_main.cpp oss_engine.cpp oss_cpu.cpp oss_kernels.hip)
target_include_directories(strata-oss PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(strata-oss PRIVATE hip::host roc::hipblas pthread)
set_source_files_properties(oss_kernels.hip PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math")
if(STRATA_OSS_NATIVE)
  set_source_files_properties(oss_cpu.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=native")
else()
  set_source_files_properties(oss_cpu.cpp PROPERTIES COMPILE_OPTIONS "-O3;-mavx2;-mfma;-mf16c")
endif()
