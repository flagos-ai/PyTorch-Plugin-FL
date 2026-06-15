# Copyright (c) 2026, BAAI. All rights reserved.
# MetaX SDK paths for native runtime + mxcc/cucc kernel builds (no cmake_metax).

if(NOT METAX_PATH)
  if(DEFINED ENV{METAX_PATH})
    set(METAX_PATH "$ENV{METAX_PATH}")
  elseif(DEFINED ENV{METAX_HOME})
    set(METAX_PATH "$ENV{METAX_HOME}")
  elseif(DEFINED ENV{MACA_PATH})
    set(METAX_PATH "$ENV{MACA_PATH}")
  elseif(DEFINED ENV{MACA_HOME})
    set(METAX_PATH "$ENV{MACA_HOME}")
  else()
    set(METAX_PATH "/opt/maca")
  endif()
endif()

set(_FLAGOS_METAX_CU_BRIDGE "${METAX_PATH}/tools/cu-bridge")
