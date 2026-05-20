# Copyright (c) 2026, BAAI. All rights reserved.
# Compile MACA device kernels with mxcc/cucc (no CMake CUDA language).

function(flagos_add_maca_kernel_objects out_var)
  if(NOT MACA_PATH)
    set(_maca_path "$ENV{MACA_PATH}")
  endif()
  if(NOT _maca_path)
    set(_maca_path "/opt/maca")
  endif()

  set(_cu_bridge "${_maca_path}/tools/cu-bridge")
  set(_maca_cc "${_cu_bridge}/bin/cucc")
  if(DEFINED ENV{MACA_MXCC})
    set(_maca_cc "$ENV{MACA_MXCC}")
  elseif(DEFINED ENV{MACA_CUCC})
    set(_maca_cc "$ENV{MACA_CUCC}")
  endif()

  if(NOT EXISTS "${_maca_cc}")
    message(FATAL_ERROR "MACA compiler not found: ${_maca_cc}")
  endif()

  set(_arch "80")
  if(DEFINED ENV{MACA_ARCH})
    set(_arch "$ENV{MACA_ARCH}")
  endif()

  set(_inc
    -I${CMAKE_SOURCE_DIR}
    -I${CMAKE_SOURCE_DIR}/csrc
    -I${CMAKE_SOURCE_DIR}/csrc/aten/backends/maca
    -I${PYTORCH_INSTALL_DIR}/include
    -I${_cu_bridge}/include
    -I${_maca_path}/include
    -I${_maca_path}/include/mcr
  )

  set(_flags -std=c++17 -fPIC -O3 -DMACA_ARCH=${_arch})

  set(_kernels add mul le all)
  set(_objs)
  set(_obj_dir "${CMAKE_CURRENT_BINARY_DIR}/maca_kernels")
  file(MAKE_DIRECTORY "${_obj_dir}")

  foreach(_k IN LISTS _kernels)
    set(_cu "${CMAKE_SOURCE_DIR}/csrc/aten/backends/maca/${_k}.cu")
    set(_obj "${_obj_dir}/${_k}.cu.o")
    add_custom_command(
      OUTPUT "${_obj}"
      COMMAND ${_maca_cc} -c ${_cu} -o ${_obj} ${_inc} ${_flags}
      DEPENDS "${_cu}"
        "${CMAKE_SOURCE_DIR}/csrc/aten/backends/maca/${_k}_kernel.cuh"
        "${CMAKE_SOURCE_DIR}/csrc/aten/backends/maca/maca_elementwise.cuh"
      COMMENT "MACA mxcc: ${_k}.cu"
      VERBATIM
    )
    list(APPEND _objs "${_obj}")
  endforeach()

  set(${out_var} ${_objs} PARENT_SCOPE)
endfunction()
