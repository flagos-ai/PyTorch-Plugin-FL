set(LIBRARY_NAME torch_fl)

option(CUDA_KERNEL "Build CUDA kernel implementations" ON)
option(FLAGGEMS_KERNEL "Build FlagGems kernel implementations" ON)
option(METAX_KERNEL "Build MetaX kernel implementations" OFF)
option(ASCEND_KERNEL "Build Ascend kernel implementations" OFF)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# Exclude runtime/accelerator (built separately as subdirectory)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/runtime/accelerator/.*")

# Exclude backend kernel directories based on options
if(NOT CUDA_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
endif()
if(NOT FLAGGEMS_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/flagos/.*")
endif()
if(NOT METAX_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/metax/.*")
endif()
if(NOT ASCEND_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/ascend/.*")
endif()

if(ACCELERATOR STREQUAL "metax")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/metax/.*\\.cu$")
endif()

set(_torch_fl_sources ${SOURCE_FILES})

if(ACCELERATOR STREQUAL "metax" AND METAX_KERNEL)
  include(${CMAKE_SOURCE_DIR}/cmake/MetaxKernels.cmake)
  flagos_add_metax_kernel_objects(_flagos_metax_kernel_objs)
  list(APPEND _torch_fl_sources ${_flagos_metax_kernel_objs})
endif()

add_library(${LIBRARY_NAME} SHARED ${_torch_fl_sources})
set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(${LIBRARY_NAME} PRIVATE
    FLAGOS_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

set(_torch_fl_link_libs torch_cpu_library flagos ${CUDA_RUNTIME_LIB})
if(FLAGGEMS_KERNEL)
  list(APPEND _torch_fl_link_libs FlagGems::operators)
endif()

if(ACCELERATOR STREQUAL "ascend")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_ASCEND=1)
  find_library(OPAPI_LIB opapi
      PATHS $ENV{ASCEND_HOME}/lib64
            /usr/local/Ascend/ascend-toolkit/latest/lib64)
  find_library(NNOPBASE_LIB nnopbase
      PATHS $ENV{ASCEND_HOME}/lib64
            /usr/local/Ascend/ascend-toolkit/latest/lib64)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
  if(OPAPI_LIB)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${OPAPI_LIB})
  endif()
  if(NNOPBASE_LIB)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${NNOPBASE_LIB})
  endif()
else()
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
endif()

target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(ACCELERATOR STREQUAL "metax")
  target_include_directories(${LIBRARY_NAME} PRIVATE
    "${METAX_PATH}/tools/cu-bridge/include"
    "${METAX_PATH}/include"
    "${METAX_PATH}/include/mcr"
    "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/metax")

  find_library(MCBLAS_LIB mcblas
      PATHS "${METAX_PATH}/lib"
            "${METAX_PATH}/lib64"
            "/opt/maca/lib"
            "/opt/maca/lib64")
  if(MCBLAS_LIB)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${MCBLAS_LIB})
  else()
    message(WARNING "mcblas not found, MetaX mm will fallback to naive kernel")
  endif()
endif()

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
