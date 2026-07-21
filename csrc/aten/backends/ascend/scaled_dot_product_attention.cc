// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include "op_preparation.h"
#include "op_api_common.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace at::native::flagos::ascend {

// Forward: _scaled_dot_product_efficient_attention(query, key, value, attn_bias,
//          compute_log_sumexp, dropout_p, is_causal, scale?)
// Returns: (output, log_sumexp, philox_seed, philox_offset)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
PrivScaledDotProductEfficientAttentionKernelAscend(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {

  namespace ascend = at::native::flagos::ascend;

  // Input validation
  TORCH_CHECK(query.dim() == 4, "query must be 4D [B, N, S, D]");
  TORCH_CHECK(query.sizes() == key.sizes(), "query and key must have same shape");
  TORCH_CHECK(query.sizes() == value.sizes(), "query and value must have same shape");
  TORCH_CHECK(query.is_privateuseone(), "SDPA Ascend: inputs must be on NPU");

  int64_t B = query.size(0);
  int64_t N = query.size(1);  // num_heads
  int64_t S = query.size(2);  // seq_len
  int64_t D = query.size(3);  // head_dim

  // Compute scale (default: 1/sqrt(D))
  double scale_value = scale.value_or(1.0 / std::sqrt(static_cast<double>(D)));

  // Dropout parameters
  double keep_prob = 1.0 - dropout_p;
  TORCH_CHECK(dropout_p == 0.0, "SDPA Ascend: dropout not yet supported (TODO)");

  // Sparse mode and token range for causal vs full attention
  int64_t sparse_mode;
  int64_t pre_tokens;
  int64_t next_tokens;
  at::Tensor atten_mask;

  if (is_causal) {
    sparse_mode = 0;  // disable sparse, use explicit mask
    pre_tokens = 65536;
    next_tokens = 65536;
    // Try inverted: true=MASK_OUT (upper triangular), false=KEEP
    auto mask_2d = at::triu(at::ones({S, S}, at::TensorOptions().dtype(at::kBool)), 1);
    atten_mask = mask_2d.unsqueeze(0).unsqueeze(0).to(query.device());  // [1,1,S,S]
  } else {
    sparse_mode = 0;  // full bidirectional
    pre_tokens = 65536;
    next_tokens = 65536;
    atten_mask = at::Tensor();  // null
  }

  // Handle attn_bias (additive bias to attention scores)
  if (attn_bias.has_value() && attn_bias->defined()) {
    TORCH_CHECK(false, "SDPA Ascend: attn_bias not yet supported (TODO: merge with atten_mask)");
  }

  // Allocate output tensors
  auto output = ascend::OpPreparation::apply_tensor_without_format(
      query.sizes().vec(), query.options());

  // softmaxMax and softmaxSum: [B, N, S, 8] float32
  auto softmax_max = ascend::OpPreparation::apply_tensor_without_format(
      {B, N, S, 8}, query.options().dtype(at::kFloat));
  auto softmax_sum = ascend::OpPreparation::apply_tensor_without_format(
      {B, N, S, 8}, query.options().dtype(at::kFloat));

  // Prepare aclnn arguments
  AclTensorWrapper q_wrap(query);
  AclTensorWrapper k_wrap(key);
  AclTensorWrapper v_wrap(value);
  AclTensorWrapper mask_wrap(is_causal ? atten_mask : at::Tensor());
  AclTensorWrapper softmax_max_wrap(softmax_max);
  AclTensorWrapper softmax_sum_wrap(softmax_sum);
  AclTensorWrapper output_wrap(output);

  char input_layout[] = "BNSD";

  // Call aclnnFlashAttentionScore
  EXEC_ASCEND_CMD(
      aclnnFlashAttentionScore,
      q_wrap.get(),
      k_wrap.get(),
      v_wrap.get(),
      nullptr,  // realShiftOptional
      nullptr,  // dropMaskOptional
      nullptr,  // paddingMaskOptional
      mask_wrap.get(),  // attenMaskOptional
      nullptr,  // prefixOptional
      scale_value,
      keep_prob,
      pre_tokens,
      next_tokens,
      N,  // headNum
      input_layout,
      1,  // innerPrecise (1 for fp16/bf16)
      sparse_mode,
      softmax_max_wrap.get(),
      softmax_sum_wrap.get(),
      nullptr,  // softmaxOutOut (not needed)
      output_wrap.get()
  );

  // Construct log_sumexp from softmaxMax and softmaxSum
  // logsumexp = log(softmaxSum) + softmaxMax
  // Shape: [B, N, S, 8] -> reduce to [B, N, S] by taking first element
  // NOTE: This is a simplification; the full [B,N,S,8] carries tiling state
  // that might be needed for exact backward. For now, take [:,:,:,0].
  at::Tensor log_sumexp;
  if (compute_log_sumexp) {
    // Move to CPU, extract [:,:,:,0], compute logsumexp, move back
    // (slice/narrow/select all unregistered on ascend backend)
    auto softmax_sum_cpu = softmax_sum.cpu();
    auto softmax_max_cpu = softmax_max.cpu();

    auto softmax_sum_0 = softmax_sum_cpu.select(3, 0);  // [B, N, S]
    auto softmax_max_0 = softmax_max_cpu.select(3, 0);  // [B, N, S]

    auto log_sumexp_cpu = at::log(softmax_sum_0) + softmax_max_0;
    log_sumexp = log_sumexp_cpu.to(query.device());
  } else {
    log_sumexp = at::empty({0}, query.options());
  }

  // philox_seed and philox_offset (for dropout RNG state)
  // Since we don't support dropout yet, return dummy tensors
  // Create on CPU first to avoid unregistered fill_ on NPU
  auto philox_seed = at::scalar_tensor(0, at::dtype(at::kLong)).to(query.device());
  auto philox_offset = at::scalar_tensor(0, at::dtype(at::kLong)).to(query.device());

  // Store softmaxMax and softmaxSum in output for backward retrieval
  // HACK: We can't modify PyTorch's autograd ctx from here, so we'll need
  // a custom autograd Function wrapper in Python or store these globally.
  // For now, return them as-is and document the limitation.
  // TODO: Implement proper autograd Function wrapper that saves both tensors.

  return std::make_tuple(output, log_sumexp, philox_seed, philox_offset);
}

// Backward: _scaled_dot_product_efficient_attention_backward(
//   grad_out, query, key, value, attn_bias, output, logsumexp,
//   philox_seed, philox_offset, dropout_p, grad_input_mask, is_causal, scale?)
// Returns: (grad_query, grad_key, grad_value, grad_attn_bias)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
PrivScaledDotProductEfficientAttentionBackwardKernelAscend(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool is_causal,
    std::optional<double> scale) {

  namespace ascend = at::native::flagos::ascend;

  TORCH_CHECK(false, "SDPA Ascend backward: NOT IMPLEMENTED YET\n"
      "CRITICAL BLOCKER: aclnnFlashAttentionScoreGrad requires softmaxMax AND softmaxSum "
      "[B,N,S,8] from forward, but PyTorch backward only passes logsumexp [B,N,S]. "
      "Cannot uniquely reconstruct both from logsumexp alone.\n"
      "SOLUTION: Need custom autograd Function that saves softmaxMax/softmaxSum in ctx, "
      "NOT a direct kernel registration. See docs/ascend_aclnn_codegen.md for details.");

  // Placeholder returns to satisfy dispatcher signature
  auto grad_query = at::empty_like(query);
  auto grad_key = at::empty_like(key);
  auto grad_value = at::empty_like(value);
  auto grad_attn_bias = at::empty({0}, query.options());

  return std::make_tuple(grad_query, grad_key, grad_value, grad_attn_bias);
}

// Register to dispatcher
REGISTER_IMPL_TO_DISPATCHER(
    PrivScaledDotProductEfficientAttentionFn,
    priv_scaled_dot_product_efficient_attention_dispatcher,
    Backend::kAscend,
    PrivScaledDotProductEfficientAttentionKernelAscend)

REGISTER_IMPL_TO_DISPATCHER(
    PrivScaledDotProductEfficientAttentionBackwardFn,
    priv_scaled_dot_product_efficient_attention_backward_dispatcher,
    Backend::kAscend,
    PrivScaledDotProductEfficientAttentionBackwardKernelAscend)

} // namespace at::native::flagos::ascend
