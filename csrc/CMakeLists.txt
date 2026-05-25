set(LIBRARY_NAME torch_fl)

option(CUDA_KERNEL "Build CUDA kernel implementations" ON)
option(FLAGGEMS_KERNEL "Build FlagGems kernel implementations" ON)
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
if(NOT ASCEND_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/ascend/.*")
endif()

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_compile_definitions(${LIBRARY_NAME} PRIVATE
    FLAGOS_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

set(_LINK_LIBS torch_cpu_library torch_python_library flagos ${CUDA_RUNTIME_LIB})
if(FLAGGEMS_KERNEL)
  list(APPEND _LINK_LIBS FlagGems::operators)
endif()

if(ACCELERATOR STREQUAL "ascend")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_ASCEND=1)
  find_library(OPAPI_LIB opapi
      PATHS $ENV{ASCEND_HOME}/lib64
            /usr/local/Ascend/ascend-toolkit/latest/lib64)
  find_library(NNOPBASE_LIB nnopbase
      PATHS $ENV{ASCEND_HOME}/lib64
            /usr/local/Ascend/ascend-toolkit/latest/lib64)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_LINK_LIBS})
  if(OPAPI_LIB)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${OPAPI_LIB})
  endif()
  if(NNOPBASE_LIB)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${NNOPBASE_LIB})
  endif()
else()
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_LINK_LIBS})
endif()

target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
