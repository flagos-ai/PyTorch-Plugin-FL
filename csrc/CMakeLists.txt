set(LIBRARY_NAME torch_fl)

option(CUDA_KERNEL "Build CUDA kernel implementations" ON)
option(FLAGGEMS_KERNEL "Build FlagGems kernel implementations" ON)
option(MACA_KERNEL "Build MACA kernel implementations" OFF)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/runtime/accelerator/.*")

if(NOT CUDA_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
endif()
if(NOT FLAGGEMS_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/flagos/.*")
endif()
if(NOT MACA_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/maca/.*")
endif()

if(ACCELERATOR STREQUAL "maca")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/maca/.*\\.cu$")
endif()

set(_torch_fl_sources ${SOURCE_FILES})

if(ACCELERATOR STREQUAL "maca" AND MACA_KERNEL)
  include(${CMAKE_SOURCE_DIR}/cmake/MacaKernels.cmake)
  flagos_add_maca_kernel_objects(_flagos_maca_kernel_objs)
  list(APPEND _torch_fl_sources ${_flagos_maca_kernel_objs})
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
target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(ACCELERATOR STREQUAL "maca")
  target_include_directories(${LIBRARY_NAME} PRIVATE
    "${MACA_PATH}/tools/cu-bridge/include"
    "${MACA_PATH}/include"
    "${MACA_PATH}/include/mcr"
    "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/maca")
endif()

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
