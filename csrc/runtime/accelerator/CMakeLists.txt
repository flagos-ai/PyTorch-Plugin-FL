cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(ACCELERATOR STREQUAL "npu")
  project(FLAGOS_RUNTIME CXX C)
elseif(ACCELERATOR STREQUAL "maca")
  project(FLAGOS_RUNTIME CXX C)
else()
  project(FLAGOS_RUNTIME CUDA CXX C)
endif()

set(LIBRARY_NAME flagos)

if(ACCELERATOR STREQUAL "npu")
  file(GLOB_RECURSE SOURCE_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/npu/csrc/*.cc"
  )

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

  find_path(CANN_INCLUDE_DIR acl/acl.h
      PATHS $ENV{ASCEND_HOME}/include
            /usr/local/Ascend/ascend-toolkit/latest/include)
  find_library(ACL_LIB ascendcl
      PATHS $ENV{ASCEND_HOME}/lib64
            /usr/local/Ascend/ascend-toolkit/latest/lib64)

  target_include_directories(${LIBRARY_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${CANN_INCLUDE_DIR})
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${ACL_LIB})
elseif(ACCELERATOR STREQUAL "maca")
  file(GLOB_RECURSE SOURCE_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/cuda/csrc/*.cc"
  )

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUDA_RUNTIME_LIB})
else()
  file(GLOB_RECURSE SOURCE_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/cuda/csrc/*.cc"
  )

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${LIBRARY_NAME} PRIVATE CUDA::cudart)
endif()

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
