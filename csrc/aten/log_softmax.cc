// Copyright (c) 2026, BAAI. All rights reserved.

#include "log_softmax.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(LogSoftmaxFn, log_softmax_dispatcher, "_log_softmax")
ADD_IMPL_TO_DISPATCHER(LogSoftmaxBackwardFn, log_softmax_backward_dispatcher, "_log_softmax_backward_data")
ADD_IMPL_TO_DISPATCHER(SoftmaxBackwardFn, softmax_backward_dispatcher, "_softmax_backward_data")

} // namespace at::native::flagos
