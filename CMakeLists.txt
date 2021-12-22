cmake_minimum_required(VERSION 3.15.2)

project(CppTrainingExamples VERSION 1.0.0)

set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)


option(MGE_WITH_JIT "Build MegEngine with JIT." ON)
option(MGE_WITH_JIT_MLIR "Build MegEngine with MLIR JIT." OFF)
option(MGE_WITH_HALIDE "Build MegEngine with Halide JIT" OFF)
option(MGE_WITH_MIDOUT_PROFILE "Build MegEngine with Midout profile." OFF)
option(MGE_WITH_MINIMUM_SIZE "Swith off MGE_ENABLE_RTTI、MGE_ENABLE_EXCEPTIONS、MGE_ENABLE_LOGGING and switch on MGE_INFERENCE_ONLY so that compile minimum load_and_run." OFF)
option(MGE_ARMV8_2_FEATURE_FP16 "Enable armv8.2-a+fp16 support" OFF)
option(MGE_DISABLE_FLOAT16 "Disable MegEngine float16 support." OFF)
option(MGE_WITH_CUDA "Enable MegEngine CUDA support." ON)
option(MGE_CUDA_USE_STATIC "Enable MegEngine CUDA static linking." ON)
option(MGE_WITH_LITE "Build MGE with lite" ON)
# BEGIN-INTERNAL
option(MGE_WITH_OPENCL "Enable MegEngine OpenCL support" OFF)
option(MGE_OPENCL_SEARCH_ALGO "OpenCL algorithm search mode for MegEngine" OFF)
option(MGE_OPENCL_SEARCH_ALGO_CHECK "Enable value checker for OpenCL algorithm search mode" OFF)
# END-INTERNAL
option(MGE_WITH_TRT "Build MegEngine with TensorRT." ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(MGE_ENABLE_RTTI "Build with RTTI" ON)
option(MGE_ENABLE_LOGGING "Build with logging" ON)
option(MGE_ENABLE_EXCEPTIONS "Build with exceptions" ON)


include(FetchContent)
set(megengine_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/MegEngine)
set(megengine_BUILD_DIR ${PROJECT_BINARY_DIR}/MegEngine)
FetchContent_Declare(MegEngine)

FetchContent_GetProperties(MegEngine)
if(NOT megengine_POPULATED)
FetchContent_Populate(MegEngine
SOURCE_DIR ${megengine_SOURCE_DIR}
BINARY_DIR ${megengine_BINARY_DIR}
)
add_subdirectory(${megengine_SOURCE_DIR} ${megengine_BINARY_DIR})
endif()

set(progressbar_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/progressbar)

file(GLOB_RECURSE SOURCES mainfile/main.cpp mnist/*.cpp helper/*.cpp ${progressbar_SOURCE_DIR}/*.cpp)

add_executable(CppTrainingExamples ${SOURCES})
target_link_libraries(CppTrainingExamples megbrain megdnn)

install(
TARGETS CppTrainingExamples 
RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)   

target_include_directories(CppTrainingExamples PRIVATE ${PROJECT_SOURCE_DIR}/third_party/MegEngine)
target_include_directories(CppTrainingExamples PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_include_directories(CppTrainingExamples PRIVATE ${PROJECT_SOURCE_DIR})