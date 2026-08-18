// Copyright (c) 2026, BAAI. All rights reserved.
//
// MUSA FlagGems hybrid registrations. Included inside
// TORCH_LIBRARY_IMPL(aten, PrivateUse1) only when FLAGGEMS_PYTHON is compiled
// and FLAGOS_USE_FLAGGEMS is enabled. Native MUSA schemas stay registered by
// musa_register.inc; this list contains only non-overlapping MThreads routes.

  m.impl("all", WrapperAll);
  m.impl("all.dims", WrapperAllDims);
  m.impl("any", WrapperAny);
  m.impl("any.dims", WrapperAnyDims);
  m.impl("index_add", WrapperIndexAdd);
  m.impl("index_add_", WrapperIndexAddInplace);
  m.impl("repeat_interleave.Tensor", WrapperRepeatInterleaveTensor);
