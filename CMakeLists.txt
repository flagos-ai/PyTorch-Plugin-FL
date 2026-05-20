cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(ACCELERATOR "cuda" CACHE STRING "Accelerator platform: cuda or maca")
option(CUDA_KERNEL "Build CUDA kernel implementations" ON)
option(FLAGGEMS_KERNEL "Build FlagGems kernel implementations" ON)
option(MACA_KERNEL "Build MACA kernel implementations" OFF)

if(ACCELERATOR STREQUAL "maca")
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MacaEnv.cmake)
  set(MACA_KERNEL ON CACHE BOOL "Build MACA kernel implementations" FORCE)
  set(CUDA_KERNEL OFF CACHE BOOL "Build CUDA kernel implementations" FORCE)
  set(FLAGGEMS_KERNEL OFF CACHE BOOL "Build FlagGems kernel implementations" FORCE)
  project(TORCH_FLAGOS CXX C)
else()
  project(TORCH_FLAGOS CUDA CXX C)
endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT ACCELERATOR STREQUAL "maca")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/lib:$ORIGIN")
elseif(WIN32)
  set(CMAKE_INSTALL_RPATH "")
endif()
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_MESSAGE NEVER)

set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)

if(ACCELERATOR STREQUAL "maca")
  # Muxi MACA SDK: use cu-bridge headers and runtime_cu library
  if(NOT MACA_PATH)
    set(MACA_PATH "$ENV{MACA_PATH}")
  endif()
  if(NOT MACA_PATH)
    set(MACA_PATH "$ENV{MACA_HOME}")
  endif()
  if(NOT MACA_PATH)
    set(MACA_PATH "/opt/maca")
  endif()
  message(STATUS "Muxi MACA SDK path: ${MACA_PATH}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MACA_PATH}/tools/cu-bridge/include;${MACA_PATH}/include;${MACA_PATH}/include/mcr"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${MACA_PATH}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE runtime_cu)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Pre-create torch::cudart so PyTorch's cuda.cmake is skipped entirely.
  # cuda.cmake guard: if(TARGET torch::cudart) return() endif()
  add_library(torch::cudart INTERFACE IMPORTED)

  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
else()
  # Workaround for CUDA 12.x where nvToolsExt is header-only (no separate library)
  # PyTorch's CMake expects CUDA::nvToolsExt target which doesn't exist in CUDA 12.x
  find_package(CUDAToolkit REQUIRED)
  if(NOT TARGET CUDA::nvToolsExt)
    # Create a dummy IMPORTED target for nvToolsExt (NVTX3 is header-only in CUDA 12.x)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    if(TARGET CUDA::nvtx3)
      set_target_properties(CUDA::nvToolsExt PROPERTIES
        INTERFACE_LINK_LIBRARIES CUDA::nvtx3)
    endif()
  endif()

  set(CUDA_RUNTIME_LIB CUDA::cudart)
endif()

find_package(Torch REQUIRED)

if(ACCELERATOR STREQUAL "maca")
  message(STATUS "MACA build: native runtime + mxcc kernels (no FlagGems)")
else()
  # Allow override via env var, otherwise try the installed flag_gems package
  if(NOT FLAGGEMS_DIR)
    set(FLAGGEMS_DIR "$ENV{FLAGGEMS_DIR}")
  endif()
  if(NOT FLAGGEMS_DIR)
    set(FLAGGEMS_DIR "/root/miniconda3/envs/pytorch/lib/python3.12/site-packages/flag_gems/lib/cmake/FlagGems")
  endif()
  # FlagGemsConfig.cmake calls find_dependency(Torch MODULE), which needs FindTorch.cmake in MODULE_PATH
  list(APPEND CMAKE_MODULE_PATH "${FLAGGEMS_DIR}")
  find_package(FlagGems REQUIRED PATHS "${FLAGGEMS_DIR}" NO_DEFAULT_PATH)

  # Embed liboperators.so dir into RPATH so it's found at runtime without LD_LIBRARY_PATH
  # FLAGGEMS_DIR is .../flag_gems/lib/cmake/FlagGems; liboperators.so is at .../flag_gems/lib/
  get_filename_component(_flaggems_libdir "${FLAGGEMS_DIR}/../.." ABSOLUTE)
  list(APPEND CMAKE_INSTALL_RPATH "${_flaggems_libdir}")
endif()

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

if(PYTORCH_INSTALL_DIR)
  list(APPEND CMAKE_INSTALL_RPATH "${PYTORCH_INSTALL_DIR}/lib")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include(${PROJECT_SOURCE_DIR}/cmake/TorchPythonTargets.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/csrc/runtime/accelerator)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_fl/csrc)
