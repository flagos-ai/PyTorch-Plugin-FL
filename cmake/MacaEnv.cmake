# Copyright (c) 2026, BAAI. All rights reserved.
# MACA SDK paths for native runtime + mxcc/cucc kernel builds (no cmake_maca).

if(NOT MACA_PATH)
  if(DEFINED ENV{MACA_PATH})
    set(MACA_PATH "$ENV{MACA_PATH}")
  elseif(DEFINED ENV{MACA_HOME})
    set(MACA_PATH "$ENV{MACA_HOME}")
  else()
    set(MACA_PATH "/opt/maca")
  endif()
endif()

set(_FLAGOS_MACA_CU_BRIDGE "${MACA_PATH}/tools/cu-bridge")
