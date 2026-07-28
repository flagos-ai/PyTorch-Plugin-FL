// Copyright (c) 2026, BAAI. All rights reserved.
//
// Fused RMSNorm forward for the Ascend backend via aclnnRmsNorm. Intercepts
// aten::_fused_rms_norm (a CompositeImplicitAutograd op force-included by
// scripts/codegen_ops.py FORCE_INCLUDE_OPS) so HF's Qwen3RMSNorm — which
// decomposes into ~6 elementwise ops + 2 dtype casts per layer — collapses to a
// single device kernel. The HF module must call F.rms_norm to route here (a
// small monkey-patch in the inference script); F.rms_norm -> aten::rms_norm ->
// aten::_fused_rms_norm -> this kernel.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

std::tuple<at::Tensor, at::Tensor> PrivFusedRmsNormKernelAscend(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight,
    std::optional<double> eps) {
  namespace ascend = at::native::flagos::ascend;

  const double epsilon = eps.value_or(1e-6);
  const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  TORCH_CHECK(norm_ndim >= 1 && input.dim() >= norm_ndim,
              "_fused_rms_norm: invalid normalized_shape for input of dim ",
              input.dim());

  // aclnnRmsNorm needs a dense input.
  at::Tensor x = input.is_contiguous() ? input : input.contiguous();

  // gamma is required by aclnn; synthesize ones matching normalized_shape when
  // weight is absent. Cast to x's dtype (aclnnRmsNorm wants matching dtypes).
  at::Tensor gamma;
  if (weight.has_value() && weight.value().defined()) {
    gamma = weight.value();
    if (gamma.scalar_type() != x.scalar_type()) {
      gamma = gamma.to(x.scalar_type());
    }
    if (!gamma.is_contiguous()) {
      gamma = gamma.contiguous();
    }
  } else {
    gamma = at::ones(normalized_shape, x.options());
  }

  at::Tensor output = ascend::OpPreparation::apply_tensor_without_format(
      x.sizes(), x.options());

  // rstd (reciprocal std) has the input's leading dims with the normalized dims
  // collapsed to 1, in float32 (CANN + torch both produce fp32 rstd). We keep
  // it for the returned tuple even though inference discards it.
  std::vector<int64_t> rstd_shape(x.sizes().begin(), x.sizes().end());
  for (int64_t i = 0; i < norm_ndim; ++i) {
    rstd_shape[x.dim() - 1 - i] = 1;
  }
  at::Tensor rstd = ascend::OpPreparation::apply_tensor_without_format(
      rstd_shape, x.options().dtype(at::kFloat));

  ascend::AclTensorWrapper acl_x(x);
  ascend::AclTensorWrapper acl_gamma(gamma);
  ascend::AclTensorWrapper acl_out(output);
  ascend::AclTensorWrapper acl_rstd(rstd);

  // aclnnRmsNorm(x, gamma, epsilon, yOut, rstdOut)
  EXEC_ASCEND_CMD(aclnnRmsNorm,
                  acl_x.get(),
                  acl_gamma.get(),
                  epsilon,
                  const_cast<aclTensor*>(acl_out.get()),
                  const_cast<aclTensor*>(acl_rstd.get()));

  return std::make_tuple(output, rstd);
}

REGISTER_IMPL_TO_DISPATCHER(
    PrivFusedRmsNormFn,
    priv_fused_rms_norm_dispatcher,
    Backend::kAscend,
    PrivFusedRmsNormKernelAscend)

} // namespace at::native::flagos
