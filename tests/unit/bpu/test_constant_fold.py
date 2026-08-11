"""Test ONNX constant folding for BPU compilation.

The constant folder must handle ORT's numpy.longlong (int64 on some platforms)
vs numpy.int64 type distinction. Without this, ConstantOfShape outputs are
mistyped as float, breaking downstream Mul/Equal/Where ops and leaving dynamic
shapes unfoldable.
"""

import onnx
import onnx.helper
import onnx.numpy_helper

from torch_fl.accelerator.bpu.constant_fold import constant_fold


def test_constantofshape_mul_where_chain_folds():
    """The regression: Expand's dynamic shape input stayed dynamic.

    Graph: ConstantOfShape → Mul → Equal → Where → (feeds Expand)
                                      ↑              ↑
                                 Constant       Constant

    Before fix: Mul failed (type mismatch float vs int64), chain broke
    After fix:  Full chain folds to [1, 32, 1]
    """
    # Build the failing subgraph extracted from Qwen2's ONNX export
    nodes = [
        # ConstantOfShape([3]) -> [1,1,1]
        onnx.helper.make_node(
            "Constant",
            [],
            ["shape_3"],
            value=onnx.helper.make_tensor("v", onnx.TensorProto.INT64, [1], [3]),
        ),
        onnx.helper.make_node(
            "ConstantOfShape",
            ["shape_3"],
            ["ones"],
            value=onnx.helper.make_tensor("v", onnx.TensorProto.INT64, [1], [1]),
        ),
        # Mul([1,1,1], -1) -> [-1,-1,-1]
        onnx.helper.make_node(
            "Constant",
            [],
            ["neg1"],
            value=onnx.helper.make_tensor("v", onnx.TensorProto.INT64, [], [-1]),
        ),
        onnx.helper.make_node("Mul", ["ones", "neg1"], ["neg_ones"]),
        # Equal([1,32,1], [-1,-1,-1]) -> [False, False, False]
        onnx.helper.make_node(
            "Constant",
            [],
            ["ref"],
            value=onnx.helper.make_tensor("v", onnx.TensorProto.INT64, [3], [1, 32, 1]),
        ),
        onnx.helper.make_node("Equal", ["ref", "neg_ones"], ["mask"]),
        # Where([False,False,False], [1,1,1], [1,32,1]) -> [1,32,1]
        onnx.helper.make_node(
            "Constant",
            [],
            ["fallback"],
            value=onnx.helper.make_tensor("v", onnx.TensorProto.INT64, [3], [1, 32, 1]),
        ),
        onnx.helper.make_node("Where", ["mask", "ones", "fallback"], ["shape"]),
    ]

    graph = onnx.helper.make_graph(
        nodes,
        "test",
        [],
        [onnx.helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, [3])],
        [],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )

    constant_fold(model)

    # The entire chain should fold: no compute nodes, output is now initializer
    assert len(model.graph.node) == 0, f"expected 0 nodes, got {len(model.graph.node)}"

    # constant_fold adds folded results as initializers but doesn't prune
    # unused ones, so we just check the final output is present
    init_names = {i.name for i in model.graph.initializer}
    assert "shape" in init_names, "output 'shape' should be an initializer"

    shape_init = next(i for i in model.graph.initializer if i.name == "shape")
    val = onnx.numpy_helper.to_array(shape_init)
    assert list(val) == [1, 32, 1]


def test_ort_int64_is_numpy_longlong():
    """Document the root cause: ORT returns numpy.longlong, not numpy.int64.

    This test will fail if ORT's behavior changes (good — tells us we can
    simplify the dtype map).
    """
    import onnxruntime as ort
    import numpy as np

    node = onnx.helper.make_node(
        "ConstantOfShape",
        ["shape"],
        ["out"],
        value=onnx.helper.make_tensor("v", onnx.TensorProto.INT64, [1], [1]),
    )
    graph = onnx.helper.make_graph(
        [node],
        "test",
        [onnx.helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, [1])],
        [onnx.helper.make_empty_tensor_value_info("out")],
        [],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    result = sess.run(["out"], {"shape": np.array([2], dtype=np.int64)})

    arr = result[0]
    assert arr.dtype.name == "int64"
    # The failing assertion if dtype.type were used in dict lookup:
    assert arr.dtype.type is not np.int64, (
        "ORT now returns np.int64 — simplify constant_fold.py dtype map"
    )
