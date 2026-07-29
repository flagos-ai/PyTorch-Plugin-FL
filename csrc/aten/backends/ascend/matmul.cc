// Copyright (c) 2026, BAAI. All rights reserved.
//
// Direct aten::matmul interception for the Ascend backend via aclnnMatmul.
//
// PyTorch's aten::matmul is CompositeImplicitAutograd and normally decomposes
// into mm + bmm + view operations before reaching PrivateUse1. torch_npu
// intercepts it at the aten::matmul level (254 aten.matmul.default/step vs
// torch_fl's mm 197 + bmm 57 + view churn 423). By registering here we
// eliminate the ~5ms/step view churn and collapse 254 ops to one aclnnMatmul
// call each, matching torch_npu's operator path exactly.
//
// Registration: register.cc TORCH_LIBRARY_IMPL(aten, PrivateUse1) adds
//   m.impl("matmul", WrapperMatmul)
// which routes to this kernel when GetBackendForOp("matmul") == kAscend;
// non-Ascend backends (MetaX etc.) fall back via ExcludeDispatchKeyGuard.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <algorithm>
#include <vector>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

// Compute the output shape for aten::matmul following NumPy/PyTorch semantics.
// aclnnMatmul handles all input dimensionalities ("for any shape mat multiply").
static std::vector<int64_t> matmul_output_shape(
    const at::Tensor& a, const at::Tensor& b) {
  int64_t da = a.dim(), db = b.dim();
  // 1-D cases
  if (da == 1 && db == 1) return {};                     // dot -> scalar
  if (da == 1 && db == 2) return {b.size(1)};            // (K,)x(K,N)->(N,)
  if (da == 2 && db == 1) return {a.size(0)};            // (M,K)x(K,)->(M,)
  if (da == 2 && db == 2) return {a.size(0), b.size(1)}; // mm
  // N-D batched: treat 1-D inputs as row/col vector, broadcast batch dims,
  // output M×N from last two dims of each input.
  auto a_sz = a.sizes().vec();
  auto b_sz = b.sizes().vec();
  bool a_1d = (da == 1), b_1d = (db == 1);
  if (a_1d) a_sz.insert(a_sz.begin(), 1);
  if (b_1d) b_sz.push_back(1);
  int64_t na = static_cast<int64_t>(a_sz.size());
  int64_t nb = static_cast<int64_t>(b_sz.size());
  int64_t n  = std::max(na, nb);
  std::vector<int64_t> out;
  for (int64_t i = 0; i < n - 2; ++i) {
    int64_t ai = na - n + i, bi = nb - n + i;
    int64_t sa = (ai >= 0) ? a_sz[ai] : 1;
    int64_t sb = (bi >= 0) ? b_sz[bi] : 1;
    out.push_back(sa == 1 ? sb : sa);
  }
  out.push_back(a_sz[na - 2]);  // M
  out.push_back(b_sz[nb - 1]);  // N
  if (a_1d) out.erase(out.end() - 2);
  if (b_1d) out.pop_back();
  return out;
}

at::Tensor MatmulKernelAscend(const at::Tensor& self,
                               const at::Tensor& other) {
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      matmul_output_shape(self, other), self.options());

  static void* opApiFuncAddr  = nullptr;
  static void* getWsFuncAddr  = nullptr;
  ascend::SigHasher hsh;
  hsh.tensor(self);
  hsh.tensor(other);
  hsh.val(cube_math_type);
  ascend::ExecAscendCached(
      "aclnnMatmul", "aclnnMatmulGetWorkspaceSize",
      opApiFuncAddr, getWsFuncAddr, hsh.h,
      {&self, &other}, {&out},
      [&](ascend::GwsFunc gws,
          std::vector<ascend::AclTensorWrapper>& in,
          std::vector<ascend::AclTensorWrapper>& out_t,
          uint64_t* pws, aclOpExecutor** pex) {
        return gws(in[0].acl_tensor, in[1].acl_tensor,
                   out_t[0].acl_tensor, cube_math_type, pws, pex);
      });
  return out;
}

} // namespace at::native::flagos
