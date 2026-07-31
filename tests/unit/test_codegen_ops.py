# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for CUDA boxing code-generation special cases."""

import pytest

from scripts import codegen_ops


_MANUAL_METADATA_OPS = {
    "_unsafe_view",
    "detach",
    "permute",
    "select.int",
    "slice.Tensor",
    "squeeze",
    "squeeze.dim",
    "transpose.int",
    "unsqueeze",
}


def test_pure_metadata_ops_are_excluded_from_generated_boxing():
    assert _MANUAL_METADATA_OPS <= codegen_ops.MANUAL_REGISTERED_OPS


_GROUP_NORM_ARGS = [
    ("const at::Tensor &", "input"),
    ("const ::std::optional<at::Tensor> &", "weight"),
    ("const ::std::optional<at::Tensor> &", "bias"),
    ("int64_t", "N"),
    ("int64_t", "C"),
    ("int64_t", "HxW"),
    ("int64_t", "group"),
    ("double", "eps"),
]


def test_native_group_norm_codegen_normalizes_input_before_boxing():
    generated = codegen_ops.gen_tuple_return(
        "native_group_norm",
        "NativeGroupNormFn",
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor>",
        _GROUP_NORM_ARGS,
    )

    contiguous = (
        "at::Tensor input_contiguous = input.is_contiguous() "
        "? input : input.contiguous();"
    )
    guard = "DeviceBoxingGuard guard(input_contiguous, weight_t, bias_t);"
    call = (
        "at::native_group_norm(input_contiguous, weight, bias, N, C, HxW, group, eps)"
    )
    assert contiguous in generated
    assert guard in generated
    assert call in generated
    assert generated.index(contiguous) < generated.index(guard) < generated.index(call)


def test_unconfigured_tuple_op_codegen_is_unchanged():
    args = [
        ("const at::Tensor &", "input"),
        ("int64_t", "dim"),
    ]
    generated = codegen_ops.gen_tuple_return(
        "example_tuple_op",
        "ExampleTupleOpFn",
        "::std::tuple<at::Tensor,at::Tensor>",
        args,
    )

    assert "input_contiguous" not in generated
    assert "DeviceBoxingGuard guard(input);" in generated
    assert "at::example_tuple_op(input, dim)" in generated


@pytest.mark.parametrize(
    "args,match",
    [
        ([("const at::Tensor &", "other")], "not in the schema"),
        ([("at::Tensor &", "input")], "must be a read-only Tensor"),
    ],
)
def test_contiguous_argument_configuration_is_validated(monkeypatch, args, match):
    monkeypatch.setitem(
        codegen_ops._CONTIGUOUS_TENSOR_ARGS_BY_OP,
        "invalid_test_op",
        ("input",),
    )

    with pytest.raises(ValueError, match=match):
        codegen_ops.prepare_contiguous_tensor_args("invalid_test_op", args)
