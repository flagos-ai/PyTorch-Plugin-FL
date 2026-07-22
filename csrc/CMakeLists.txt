set(LIBRARY_NAME torch_fl)

option(CUDA_KERNEL "Build CUDA kernel implementations" ON)
option(FLAGGEMS_KERNEL "Build FlagGems C++ kernel wrappers (requires liboperators.so)" ON)
option(FLAGGEMS_PYTHON "Build FlagGems Python kernel wrappers (calls via Python, no C++ linking)" OFF)
option(METAX_KERNEL "Build MetaX kernel implementations" OFF)
option(ASCEND_KERNEL "Build Ascend kernel implementations" OFF)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# Exclude runtime/accelerator (built separately as subdirectory)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/runtime/accelerator/.*")
# stream_api is built as a separate shared lib
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/runtime/accelerator/ascend/stream_api\\.cc")

# Exclude backend kernel directories based on options
if(NOT CUDA_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
endif()
if(NOT FLAGGEMS_KERNEL AND NOT FLAGGEMS_PYTHON)
  # Neither C++ nor Python wrappers: exclude entire flagos backend
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/flagos/.*")
elseif(FLAGGEMS_PYTHON AND NOT FLAGGEMS_KERNEL)
  # Python wrappers only: exclude C++ wrappers (bmm.cc, cat.cc, etc.) but keep python_op_caller and python_wrapper/
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/flagos/(bmm|cat|embedding|mm|softmax)\\.cc")
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

# Build stream_api as a standalone shared lib so liboperators.so can resolve it at runtime
if(FLAGGEMS_KERNEL AND ACCELERATOR STREQUAL "ascend")
  add_library(stream_api SHARED "${CMAKE_CURRENT_SOURCE_DIR}/runtime/accelerator/ascend/stream_api.cc")
  target_compile_definitions(stream_api PRIVATE USE_ASCEND=1)
  target_include_directories(stream_api PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${ASCEND_HOME}/include
  )
  find_library(ASCENDCL_LIB ascendcl
      PATHS ${ASCEND_HOME}/lib64)
  target_link_libraries(stream_api PRIVATE torch_cpu_library ${ASCENDCL_LIB})
  install(TARGETS stream_api
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

target_compile_definitions(${LIBRARY_NAME} PRIVATE
    FLAGOS_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

set(_torch_fl_link_libs torch_cpu_library torch_python_library flagos ${CUDA_RUNTIME_LIB})
if(FLAGGEMS_KERNEL)
  if(TARGET FlagGems::operators)
    list(APPEND _torch_fl_link_libs FlagGems::operators)
  elseif(TARGET operators)
    list(APPEND _torch_fl_link_libs operators)
  endif()
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
elseif(ACCELERATOR STREQUAL "tsingmicro")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_TSINGMICRO=1)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
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
