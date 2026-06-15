# Copyright (c) 2026, BAAI. All rights reserved.
# Compile MetaX device kernels with mxcc/cucc (no CMake CUDA language).

function(flagos_add_metax_kernel_objects out_var)
  if(NOT METAX_PATH)
    set(_metax_path "$ENV{METAX_PATH}")
  endif()
  if(NOT _metax_path)
    if(DEFINED ENV{MACA_PATH})
      set(_metax_path "$ENV{MACA_PATH}")
    elseif(DEFINED ENV{MACA_HOME})
      set(_metax_path "$ENV{MACA_HOME}")
    else()
      set(_metax_path "/opt/maca")
    endif()
  endif()

  set(_cu_bridge "${_metax_path}/tools/cu-bridge")
  set(_metax_cc "${_cu_bridge}/bin/cucc")
  if(DEFINED ENV{METAX_MXCC})
    set(_metax_cc "$ENV{METAX_MXCC}")
  elseif(DEFINED ENV{METAX_CUCC})
    set(_metax_cc "$ENV{METAX_CUCC}")
  endif()

  if(NOT EXISTS "${_metax_cc}")
    message(FATAL_ERROR "MetaX compiler not found: ${_metax_cc}")
  endif()

  set(_arch "80")
  if(DEFINED ENV{METAX_ARCH})
    set(_arch "$ENV{METAX_ARCH}")
  elseif(DEFINED ENV{MACA_ARCH})
    set(_arch "$ENV{MACA_ARCH}")
  endif()

  set(_inc
    -I${CMAKE_SOURCE_DIR}
    -I${CMAKE_SOURCE_DIR}/csrc
    -I${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax
    -I${PYTORCH_INSTALL_DIR}/include
    -I${_cu_bridge}/include
    -I${_metax_path}/include
    -I${_metax_path}/include/mcr
  )

  set(_flags -std=c++17 -fPIC -O3 -DMETAX_ARCH=${_arch})

  set(_kernels add mul le all neg embedding mean mm bmm cos sin rsqrt silu pow sum ones_like softmax where bitwise_and)
  set(_objs)
  set(_obj_dir "${CMAKE_CURRENT_BINARY_DIR}/metax_kernels")
  file(MAKE_DIRECTORY "${_obj_dir}")

  foreach(_k IN LISTS _kernels)
    set(_cu "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax/${_k}.cu")
    set(_obj "${_obj_dir}/${_k}.cu.o")
    add_custom_command(
      OUTPUT "${_obj}"
      COMMAND ${_metax_cc} -c ${_cu} -o ${_obj} ${_inc} ${_flags}
      DEPENDS "${_cu}"
        "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax/${_k}_kernel.cuh"
        "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax/metax_elementwise.cuh"
      COMMENT "MetaX mxcc: ${_k}.cu"
      VERBATIM
    )
    list(APPEND _objs "${_obj}")
  endforeach()

  set(${out_var} ${_objs} PARENT_SCOPE)
endfunction()
