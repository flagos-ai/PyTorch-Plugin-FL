cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(ACCELERATOR "cuda" CACHE STRING "Accelerator platform: cuda, maca, or ascend")

# Kernel build options (can be overridden via -D flags from setup.py)
option(CUDA_KERNEL "Build CUDA kernel implementations" ON)
option(FLAGGEMS_KERNEL "Build FlagGems C++ kernel wrappers (requires liboperators.so)" ON)
option(FLAGGEMS_PYTHON "Build FlagGems Python kernel wrappers (calls via Python, no C++ linking)" OFF)
option(ASCEND_KERNEL "Build Ascend kernel implementations" OFF)

if(ACCELERATOR STREQUAL "ascend")
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "maca")
  project(TORCH_FLAGOS CXX C)
else()
  project(TORCH_FLAGOS CUDA CXX C)
endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

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

if(ACCELERATOR STREQUAL "ascend")
  # Huawei CANN SDK
  if(NOT ASCEND_HOME)
    set(ASCEND_HOME "$ENV{ASCEND_HOME}")
  endif()
  if(NOT ASCEND_HOME)
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
  message(STATUS "Huawei CANN SDK path: ${ASCEND_HOME}")

  set(ASCEND_KERNEL ON CACHE BOOL "Build Ascend kernel implementations" FORCE)
  set(CUDA_KERNEL OFF CACHE BOOL "Build CUDA kernel implementations" FORCE)

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_HOME}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${ASCEND_HOME}/lib64")
  target_link_libraries(cuda_runtime_compat INTERFACE ascendcl)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(ACCELERATOR STREQUAL "maca")
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

if(FLAGGEMS_KERNEL)
  # Allow override via cmake var or env var
  # setup.py passes -DFlagGems_DIR=..., CMakeLists uses FLAGGEMS_DIR internally
  if(FlagGems_DIR AND NOT FLAGGEMS_DIR)
    set(FLAGGEMS_DIR "${FlagGems_DIR}")
  endif()
  if(NOT FLAGGEMS_DIR)
    set(FLAGGEMS_DIR "$ENV{FLAGGEMS_DIR}")
  endif()
  if(NOT FLAGGEMS_DIR)
    # Try FLAGGEMS_SOURCE_DIR (cmake var from setup.py or env var)
    if(NOT FLAGGEMS_SOURCE_DIR)
      set(FLAGGEMS_SOURCE_DIR "$ENV{FLAGGEMS_SOURCE_DIR}")
    endif()
    if(FLAGGEMS_SOURCE_DIR)
      # Check if it points to the repo root (has build/ dir with FlagGemsConfig.cmake)
      file(GLOB _FLAGGEMS_BUILD_CONFIGS "${FLAGGEMS_SOURCE_DIR}/build/*/FlagGemsConfig.cmake")
      if(_FLAGGEMS_BUILD_CONFIGS)
        list(GET _FLAGGEMS_BUILD_CONFIGS 0 _FLAGGEMS_CONFIG)
        get_filename_component(FLAGGEMS_DIR "${_FLAGGEMS_CONFIG}" DIRECTORY)
      endif()
    endif()
  endif()
  if(NOT FLAGGEMS_DIR)
    # Auto-detect from installed flag_gems package
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import flag_gems; print(flag_gems.__path__[0])"
      OUTPUT_VARIABLE _FLAGGEMS_PKG_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _FLAGGEMS_DETECT_RET
    )
    if(_FLAGGEMS_DETECT_RET EQUAL 0 AND _FLAGGEMS_PKG_PATH)
      set(FLAGGEMS_DIR "${_FLAGGEMS_PKG_PATH}/lib/cmake/FlagGems")
    endif()
  endif()
  if(NOT FLAGGEMS_DIR)
    message(FATAL_ERROR "Cannot find FlagGems. Set FLAGGEMS_DIR or FLAGGEMS_SOURCE_DIR.")
  endif()
  message(STATUS "FlagGems cmake dir: ${FLAGGEMS_DIR}")
  # FlagGemsConfig.cmake calls find_dependency(Torch MODULE), which needs FindTorch.cmake in MODULE_PATH
  if(FLAGGEMS_SOURCE_DIR AND EXISTS "${FLAGGEMS_SOURCE_DIR}/cmake")
    list(APPEND CMAKE_MODULE_PATH "${FLAGGEMS_SOURCE_DIR}/cmake")
  endif()
  list(APPEND CMAKE_MODULE_PATH "${FLAGGEMS_DIR}")
  find_package(FlagGems REQUIRED PATHS "${FLAGGEMS_DIR}" NO_DEFAULT_PATH)

  # Embed liboperators.so dir into RPATH so it's found at runtime without LD_LIBRARY_PATH
  set(_flaggems_libdir "${FLAGGEMS_DIR}/lib")
  get_target_property(_flaggems_lib_location operators IMPORTED_LOCATION_RELEASE)
  if(NOT _flaggems_lib_location)
    get_target_property(_flaggems_lib_location operators IMPORTED_LOCATION)
  endif()
  if(_flaggems_lib_location)
    get_filename_component(_flaggems_libdir "${_flaggems_lib_location}" DIRECTORY)
  endif()
  message(STATUS "FlagGems liboperators RPATH: ${_flaggems_libdir}")
  list(APPEND CMAKE_INSTALL_RPATH "${_flaggems_libdir}")
endif()

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include(${PROJECT_SOURCE_DIR}/cmake/TorchPythonTargets.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/csrc/runtime/accelerator)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_fl/csrc)
