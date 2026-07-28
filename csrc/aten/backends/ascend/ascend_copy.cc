// Copyright (c) 2026, BAAI. All rights reserved.

#include "ascend_copy.h"

#include <ATen/core/Tensor.h>
#include "op_api_common.h"

namespace at::native::flagos::ascend {

bool StridedCopy(const at::Tensor& dst, const at::Tensor& src) {
  if (!dst.defined() || !src.defined()) {
    return false;
  }
  if (!dst.is_privateuseone() || !src.is_privateuseone()) {
    return false;
  }
  if (dst.numel() == 0) {
    return true;  // nothing to copy
  }

  // aclnnInplaceCopy(selfRef, src): writes src into selfRef, honoring the
  // strides/offset recorded on each aclTensor. AclTensorWrapper preserves the
  // tensor's sizes/strides/offset, so a non-contiguous src is copied correctly
  // into the (contiguous) dst without a host round-trip.
  AclTensorWrapper dst_wrap(dst);
  AclTensorWrapper src_wrap(src);
  EXEC_ASCEND_CMD(aclnnInplaceCopy,
                  const_cast<aclTensor*>(dst_wrap.get()),
                  src_wrap.get());
  return true;
}

at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype) {
  if (!src.defined() || !src.is_privateuseone()) {
    return {};
  }
  // aclnnCast expects a dense input; make src contiguous first (cheap, and the
  // callers in _to_copy already pass a contiguous tensor).
  at::Tensor src_c = src.is_contiguous() ? src : src.contiguous();
  at::Tensor out = at::empty(src_c.sizes(), src_c.options().dtype(dtype));
  if (src_c.numel() == 0) {
    return out;
  }

  // aclnnCast(self, dtype, out): converts self to the given aclDataType
  // on-device. Route through the repeatable-executor cache: RMSNorm emits two
  // fp16<->fp32 casts per layer (285/step) at fixed decode shapes, so the
  // GetWorkspaceSize + aclCreateTensor build cost is paid once per shape. The
  // target aclDataType is baked into the executor at build time, so it must be
  // part of the cache key (folded in via SigHasher::val below).
  const aclDataType acl_dtype = ToAclDataType(dtype);
  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  SigHasher hsh;
  hsh.tensor(src_c);
  hsh.tensor(out);
  hsh.val(static_cast<int32_t>(acl_dtype));
  ExecAscendCached(
      "aclnnCast", "aclnnCastGetWorkspaceSize",
      opApiFuncAddr, getWsFuncAddr, hsh.h,
      {&src_c}, {&out},
      [&](GwsFunc gws,
          std::vector<AclTensorWrapper>& in,
          std::vector<AclTensorWrapper>& out_t,
          uint64_t* pws, aclOpExecutor** pex) {
        return gws(in[0].acl_tensor, acl_dtype, out_t[0].acl_tensor, pws, pex);
      });
  return out;
}

} // namespace at::native::flagos::ascend
