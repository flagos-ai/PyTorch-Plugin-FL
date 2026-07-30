# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(ACCELERATOR "cuda" CACHE STRING "Accelerator platform: cuda, metax, ascend, tsingmicro, dcu, gcu, or musa")

# Kernel build options (can be overridden via -D flags from setup.py)
option(CUDA_KERNEL "Build CUDA kernel implementations" ON)
option(FLAGGEMS_KERNEL "Build FlagGems C++ kernel wrappers (requires liboperators.so)" ON)
option(FLAGGEMS_PYTHON "Build FlagGems Python kernel wrappers (calls via Python, no C++ linking)" ON)
option(METAX_KERNEL "Build MetaX kernel implementations" OFF)
option(ASCEND_KERNEL "Build Ascend kernel implementations" OFF)
option(GCU_KERNEL "Build Enflame GCU (topsaten) kernel implementations" OFF)
option(MUSA_KERNEL "Build Moore Threads MUSA kernel implementations (mudnn)" OFF)

if(ACCELERATOR STREQUAL "ascend")
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "metax")
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MetaxEnv.cmake)
  # Boxing mode ($ENV{FLAGOS_METAX_BOXING}=1): reuse the generated CUDA boxing
  # kernels (csrc/aten/generated/cuda_kernels.cc, host g++, no mxcc) that
  # dispatch PrivateUse1 -> CUDA into maca's libtorch_cuda.so, instead of the
  # hand-written mxcc .cu kernels under backends/metax/. Keeps the MetaX SDK
  # runtime + cu-bridge headers path (no nvcc / project(...CUDA)).
  if(DEFINED ENV{FLAGOS_METAX_BOXING} AND NOT "$ENV{FLAGOS_METAX_BOXING}" STREQUAL "0")
    set(FLAGOS_METAX_BOXING ON)
  endif()
  if(FLAGOS_METAX_BOXING)
    set(METAX_KERNEL OFF CACHE BOOL "Build MetaX kernel implementations" FORCE)
  else()
    set(METAX_KERNEL ON CACHE BOOL "Build MetaX kernel implementations" FORCE)
  endif()
  set(CUDA_KERNEL OFF CACHE BOOL "Build CUDA kernel implementations" FORCE)
  set(FLAGGEMS_KERNEL OFF CACHE BOOL "Build FlagGems kernel implementations" FORCE)
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "tsingmicro")
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "dcu")
  # Hygon DCU (DTK). The vendor torch wheel is a hipified build that registers
  # its HIP kernels under the CUDA dispatch key (tensors report device type
  # kCUDA), so the generated PrivateUse1 -> CUDA boxing kernels dispatch into
  # libtorch_hip.so unchanged. DTK also ships a CUDA compatibility toolkit whose
  # libcudart.so.12 is a thin shim over libgalaxyhip.so, so the runtime sources
  # under csrc/runtime/accelerator/cuda/ compile as-is against host g++.
  # No nvcc and no CUDA language: nothing outside backends/metax/ is a .cu.
  set(CUDA_KERNEL OFF CACHE BOOL "Build CUDA kernel implementations" FORCE)
  set(FLAGGEMS_KERNEL OFF CACHE BOOL "Build FlagGems kernel implementations" FORCE)
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "gcu")
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "musa")
  # Moore Threads MUSA takes the native operator route (like Ascend/GCU), not
  # CUDA boxing: the MUSA toolkit ships no CUDA runtime, so there is no vendor
  # dispatch key to box into. The generated kernels call mudnn -- the vendor's
  # torch-independent kernel library -- directly.
  set(MUSA_KERNEL ON CACHE BOOL "Build MUSA kernel implementations" FORCE)
  set(CUDA_KERNEL OFF CACHE BOOL "Build CUDA kernel implementations" FORCE)
  set(FLAGGEMS_KERNEL OFF CACHE BOOL "Build FlagGems kernel implementations" FORCE)
  project(TORCH_FLAGOS CXX C)
else()
  # project(... CUDA) runs nvcc compiler-id before find_package(CUDAToolkit); conda
  # layouts need CUDAToolkit_ROOT (headers under targets/x86_64-linux/include).
  if(NOT CUDAToolkit_ROOT)
    if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "CUDA toolkit root")
    elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/targets/x86_64-linux/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}/targets/x86_64-linux" CACHE PATH "CUDA toolkit root")
    endif()
  endif()
  if(NOT CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc" CACHE FILEPATH "")
    elseif(CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "")
    endif()
  endif()
  project(TORCH_FLAGOS CUDA CXX C)
endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT ACCELERATOR STREQUAL "metax" AND NOT ACCELERATOR STREQUAL "ascend"
   AND NOT ACCELERATOR STREQUAL "tsingmicro" AND NOT ACCELERATOR STREQUAL "dcu"
   AND NOT ACCELERATOR STREQUAL "gcu" AND NOT ACCELERATOR STREQUAL "musa")
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
elseif(ACCELERATOR STREQUAL "tsingmicro")
  # TsingMicro Kuiper SDK
  set(KUIPER_ROOT "/usr/local/kuiper" CACHE PATH "Kuiper SDK root")
  message(STATUS "TsingMicro Kuiper SDK path: ${KUIPER_ROOT}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${KUIPER_ROOT}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${KUIPER_ROOT}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE hpgr)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed Kuiper library path in RPATH so LD_LIBRARY_PATH is not needed at runtime.
  list(APPEND CMAKE_INSTALL_RPATH "${KUIPER_ROOT}/lib")

  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(ACCELERATOR STREQUAL "dcu")
  # Hygon DTK. DTK_ROOT defaults to /opt/dtk (ROCM_PATH is what env.sh exports).
  if(NOT DTK_ROOT)
    set(DTK_ROOT "$ENV{DTK_ROOT}")
  endif()
  if(NOT DTK_ROOT)
    set(DTK_ROOT "$ENV{ROCM_PATH}")
  endif()
  if(NOT DTK_ROOT)
    set(DTK_ROOT "/opt/dtk")
  endif()
  message(STATUS "Hygon DTK path: ${DTK_ROOT}")

  # DTK's CUDA compatibility toolkit. Its libcudart.so.12 is a shim over
  # libgalaxyhip.so -- the same runtime libtorch_hip.so uses -- so there is one
  # driver state, not two.
  file(GLOB _DCU_CUDA_CANDIDATES "${DTK_ROOT}/cuda/cuda-*")
  set(DCU_CUDA_ROOT "")
  foreach(_cand IN LISTS _DCU_CUDA_CANDIDATES)
    if(EXISTS "${_cand}/include/cuda_runtime.h")
      set(DCU_CUDA_ROOT "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT DCU_CUDA_ROOT)
    message(FATAL_ERROR
        "ACCELERATOR=dcu selected, but the DTK CUDA compatibility toolkit was "
        "not found under ${DTK_ROOT}/cuda/. Set DTK_ROOT (or ROCM_PATH) to a "
        "DTK installation that provides cuda-*/include/cuda_runtime.h.")
  endif()
  message(STATUS "DTK CUDA compatibility toolkit: ${DCU_CUDA_ROOT}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${DCU_CUDA_ROOT}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${DCU_CUDA_ROOT}/lib64")
  target_link_libraries(cuda_runtime_compat INTERFACE cudart)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed the compat + DTK library paths so LD_LIBRARY_PATH is not needed.
  list(APPEND CMAKE_INSTALL_RPATH "${DCU_CUDA_ROOT}/lib64" "${DTK_ROOT}/lib")

  # Pre-create torch::cudart so PyTorch's cuda.cmake is skipped entirely
  # (guard: if(TARGET torch::cudart) return()).
elseif(ACCELERATOR STREQUAL "gcu")
  # Enflame TopsRider SDK: tops runtime (libtopsrt) + topsaten operator library.
  if(NOT TOPS_ROOT)
    set(TOPS_ROOT "$ENV{TOPS_HOME}")
  endif()
  if(NOT TOPS_ROOT)
    set(TOPS_ROOT "/opt/tops")
  endif()
  message(STATUS "Enflame TopsRider SDK path: ${TOPS_ROOT}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TOPS_ROOT}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${TOPS_ROOT}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE topsrt)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed the tops library path in RPATH so LD_LIBRARY_PATH is not needed at runtime.
  list(APPEND CMAKE_INSTALL_RPATH "${TOPS_ROOT}/lib")

  # topsaten (the aten-style op library) ships in the distro paths, not under
  # ${TOPS_ROOT}/lib; its headers live in ${TOPS_ROOT}/include/gcu.
  if(GCU_KERNEL)
    find_library(TOPSATEN_LIB topsaten
        PATHS "${TOPS_ROOT}/lib" /usr/lib /usr/lib64)
    if(NOT TOPSATEN_LIB)
      message(FATAL_ERROR
        "GCU_KERNEL=ON but libtopsaten.so was not found (searched "
        "${TOPS_ROOT}/lib, /usr/lib, /usr/lib64). Install the TopsRider "
        "topsaten package, or build with -DGCU_KERNEL=OFF to fall back to CPU "
        "for all compute ops.")
    endif()
    find_path(TOPSATEN_INCLUDE_DIR topsaten/topsaten.h
        PATHS "${TOPS_ROOT}/include/gcu" /usr/include/gcu)
    if(NOT TOPSATEN_INCLUDE_DIR)
      message(FATAL_ERROR
        "GCU_KERNEL=ON but topsaten/topsaten.h was not found (searched "
        "${TOPS_ROOT}/include/gcu, /usr/include/gcu).")
    endif()
    set(TOPSATEN_INCLUDE_DIRS "${TOPSATEN_INCLUDE_DIR}" "${TOPS_ROOT}/include")
    message(STATUS "topsaten library: ${TOPSATEN_LIB}")
    message(STATUS "topsaten include: ${TOPSATEN_INCLUDE_DIR}")
    get_filename_component(_topsaten_libdir "${TOPSATEN_LIB}" DIRECTORY)
    list(APPEND CMAKE_INSTALL_RPATH "${_topsaten_libdir}")
  endif()

  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(ACCELERATOR STREQUAL "musa")
  # Moore Threads MUSA toolkit: musart runtime. There is no libcudart shim here
  # (unlike MetaX/DCU), so nothing pretends to be CUDA -- cuda_runtime_compat
  # points at the native musa* headers and libmusart, and the runtime sources
  # under csrc/runtime/accelerator/musa/ call musaMalloc/musaStreamCreate/etc.
  if(NOT MUSA_HOME)
    set(MUSA_HOME "$ENV{MUSA_HOME}")
  endif()
  if(NOT MUSA_HOME)
    set(MUSA_HOME "/usr/local/musa")
  endif()
  if(NOT EXISTS "${MUSA_HOME}/include/musa_runtime.h")
    message(FATAL_ERROR
        "ACCELERATOR=musa selected, but the MUSA toolkit was not found "
        "(no ${MUSA_HOME}/include/musa_runtime.h). Set MUSA_HOME to a valid "
        "MUSA installation.")
  endif()
  message(STATUS "Moore Threads MUSA toolkit path: ${MUSA_HOME}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MUSA_HOME}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${MUSA_HOME}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE musart)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed the MUSA library path in RPATH so LD_LIBRARY_PATH is not needed.
  list(APPEND CMAKE_INSTALL_RPATH "${MUSA_HOME}/lib")

  # Compute ops go through mudnn (libmudnn.so), the vendor's kernel library.
  # It links against musart only and pulls in no torch symbols, so it ships in
  # the MUSA toolkit next to musart and needs no separate root: no torch_musa
  # package, no extracted tree, no version pinning against the active torch.
  #
  # (The earlier route called torch_musa's flat at::musa::* API from
  # libmusa_python.so. That library links against torch and embeds its C++
  # object layout, which pinned the plugin to one exact torch build --
  # sizeof(c10::MessageLogger) changed 408 -> 400 between 2.9.1 and 2.10 and
  # corrupted the vendor .so's stack. mudnn has no such coupling.)
  if(MUSA_KERNEL)
    find_library(MUDNN_LIB mudnn PATHS "${MUSA_HOME}/lib" NO_DEFAULT_PATH)
    if(NOT MUDNN_LIB)
      message(FATAL_ERROR
          "MUSA_KERNEL=ON but libmudnn.so was not found under ${MUSA_HOME}/lib. "
          "Install the mudnn component of the MUSA toolkit, or build with "
          "-DMUSA_KERNEL=OFF to fall back to CPU for compute ops.")
    endif()
    if(NOT EXISTS "${MUSA_HOME}/include/mudnncxx/mudnn.h")
      message(FATAL_ERROR
          "MUSA_KERNEL=ON but the mudnn C++ headers were not found "
          "(no ${MUSA_HOME}/include/mudnncxx/mudnn.h).")
    endif()
    message(STATUS "mudnn op library: ${MUDNN_LIB}")
  endif()

  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(ACCELERATOR STREQUAL "metax")
  # MetaX SDK: use cu-bridge headers and runtime_cu library
  if(NOT METAX_PATH)
    set(METAX_PATH "$ENV{METAX_PATH}")
  endif()
  if(NOT METAX_PATH)
    set(METAX_PATH "$ENV{METAX_HOME}")
  endif()
  if(NOT METAX_PATH)
    set(METAX_PATH "$ENV{MACA_PATH}")
  endif()
  if(NOT METAX_PATH)
    set(METAX_PATH "$ENV{MACA_HOME}")
  endif()
  if(NOT METAX_PATH)
    set(METAX_PATH "/opt/maca")
  endif()
  message(STATUS "MetaX SDK path: ${METAX_PATH}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${METAX_PATH}/tools/cu-bridge/include;${METAX_PATH}/include;${METAX_PATH}/include/mcr"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${METAX_PATH}/lib")
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

if(ACCELERATOR STREQUAL "dcu")
  # DTK exports MIOpen with a build-machine absolute path baked into its link
  # interface: /usr/lib/x86_64-linux-gnu/librt.so. glibc >= 2.34 folded librt
  # into libc, so current distros ship only librt.a + librt.so.1 -- that .so
  # symlink no longer exists and the generated makefiles die with
  # "No rule to make target '/usr/lib/x86_64-linux-gnu/librt.so'".
  # Rewrite any dangling absolute path to its bare -l name; the linker resolves
  # `rt` against the stub archive. Nothing to do on hosts where the path exists.
  foreach(_dcu_target MIOpen)
    if(TARGET ${_dcu_target})
      get_target_property(_dcu_libs ${_dcu_target} INTERFACE_LINK_LIBRARIES)
      if(_dcu_libs)
        set(_dcu_fixed_libs "")
        foreach(_dcu_lib IN LISTS _dcu_libs)
          if(IS_ABSOLUTE "${_dcu_lib}" AND NOT EXISTS "${_dcu_lib}")
            get_filename_component(_dcu_name "${_dcu_lib}" NAME_WE)
            string(REGEX REPLACE "^lib" "" _dcu_name "${_dcu_name}")
            message(STATUS
                "DCU: ${_dcu_target} links missing ${_dcu_lib}, using -l${_dcu_name}")
            list(APPEND _dcu_fixed_libs "${_dcu_name}")
          else()
            list(APPEND _dcu_fixed_libs "${_dcu_lib}")
          endif()
        endforeach()
        set_target_properties(${_dcu_target} PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_dcu_fixed_libs}")
      endif()
    endif()
  endforeach()
endif()

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
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  if(FLAGGEMS_SOURCE_DIR AND EXISTS "${FLAGGEMS_SOURCE_DIR}/cmake")
    list(APPEND CMAKE_MODULE_PATH "${FLAGGEMS_SOURCE_DIR}/cmake")
  endif()
  list(APPEND CMAKE_MODULE_PATH "${FLAGGEMS_DIR}")
  find_package(FlagGems REQUIRED PATHS "${FLAGGEMS_DIR}" NO_DEFAULT_PATH)

  # Embed liboperators.so dir into RPATH so it's found at runtime without LD_LIBRARY_PATH
  if(TARGET FlagGems::operators)
    set(_flaggems_operators_target FlagGems::operators)
  elseif(TARGET operators)
    set(_flaggems_operators_target operators)
  else()
    message(FATAL_ERROR "FlagGems package does not export an operators target")
  endif()

  get_filename_component(_flaggems_libdir "${FLAGGEMS_DIR}/../.." ABSOLUTE)
  get_target_property(
    _flaggems_lib_location
    ${_flaggems_operators_target}
    IMPORTED_LOCATION_RELEASE
  )
  if(NOT _flaggems_lib_location)
    get_target_property(
      _flaggems_lib_location
      ${_flaggems_operators_target}
      IMPORTED_LOCATION
    )
  endif()
  if(_flaggems_lib_location)
    get_filename_component(_flaggems_libdir "${_flaggems_lib_location}" DIRECTORY)
  endif()
  message(STATUS "FlagGems liboperators RPATH: ${_flaggems_libdir}")
  list(APPEND CMAKE_INSTALL_RPATH "${_flaggems_libdir}")
elseif(ACCELERATOR STREQUAL "metax")
  if(FLAGOS_METAX_BOXING)
    message(STATUS "MetaX build: native runtime + CUDA boxing kernels (host g++, no mxcc/FlagGems)")
  else()
    message(STATUS "MetaX build: native runtime + mxcc kernels (no FlagGems)")
  endif()
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
