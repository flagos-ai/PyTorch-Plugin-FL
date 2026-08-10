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

"""ONNX constant folding via iterative evaluation.

hbdk4 shape inference rejects graphs with dynamic shape computations (Expand
whose shape comes from a Where chain, attention mask construction). Those
computations are constant given fixed input shapes, so folding them into
initializers makes the graph acceptable.
"""

from __future__ import annotations

import logging


log = logging.getLogger("torch_fl.bpu")


def constant_fold(model) -> None:
    """Fold constant subgraphs in place by evaluating them with onnxruntime.

    Iterates until no new constants are found (max 10 rounds). A node is
    foldable when all its inputs are initializers or outputs of already-folded
    nodes. Folded outputs are added as initializers and their producer nodes
    are removed.

    Args:
        model: onnx.ModelProto, modified in place.
    """
    import onnx
    import onnxruntime as ort

    const_vals = {
        init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer
    }

    # Also extract Constant nodes (not yet in initializers)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const_vals[node.output[0]] = onnx.numpy_helper.to_array(attr.t)

    total_folded = 0
    for iteration in range(10):
        folded_this_iter = 0
        for node in model.graph.node:
            # Already folded
            if node.output[0] in const_vals:
                continue

            # Not all inputs are constant
            if not all(inp in const_vals or not inp for inp in node.input):
                continue

            # Try to evaluate this node in isolation
            try:
                inputs_vi = []
                feed = {}
                for inp in node.input:
                    if inp:
                        val = const_vals[inp]
                        feed[inp] = val
                        # Map numpy dtype to ONNX TensorProto type. Use dtype.name
                        # instead of dtype.type because ORT returns numpy.longlong
                        # for int64, which doesn't match np.int64 in dict lookup.
                        dtype_name_map = {
                            "float16": onnx.TensorProto.FLOAT16,
                            "float32": onnx.TensorProto.FLOAT,
                            "float64": onnx.TensorProto.DOUBLE,
                            "int8": onnx.TensorProto.INT8,
                            "int16": onnx.TensorProto.INT16,
                            "int32": onnx.TensorProto.INT32,
                            "int64": onnx.TensorProto.INT64,
                            "uint8": onnx.TensorProto.UINT8,
                            "bool": onnx.TensorProto.BOOL,
                        }
                        dtype = dtype_name_map.get(
                            val.dtype.name, onnx.TensorProto.FLOAT
                        )
                        inputs_vi.append(
                            onnx.helper.make_tensor_value_info(
                                inp, dtype, list(val.shape)
                            )
                        )

                outputs_vi = [
                    onnx.helper.make_empty_tensor_value_info(o) for o in node.output
                ]

                g = onnx.helper.make_graph([node], "fold", inputs_vi, outputs_vi, [])
                m = onnx.helper.make_model(
                    g,
                    opset_imports=model.opset_import,
                    ir_version=model.ir_version,
                )

                sess = ort.InferenceSession(
                    m.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                results = sess.run(list(node.output), feed)

                for name, val in zip(node.output, results):
                    const_vals[name] = val
                    folded_this_iter += 1

            except Exception as e:
                # Node can't be folded (ORT doesn't support it, or attributes
                # are missing, etc). Skip silently — the graph may still compile.
                if node.op_type in ("ConstantOfShape", "Mul", "Equal", "Where"):
                    log.debug(
                        "constant_fold: %s -> %s failed: %s",
                        node.op_type,
                        node.output[0],
                        str(e),
                    )
                pass

        total_folded += folded_this_iter
        if folded_this_iter == 0:
            break

    if total_folded == 0:
        return

    log.debug(
        "constant_fold: folded %d node(s) in %d iteration(s)",
        total_folded,
        iteration + 1,
    )

    # Remove folded nodes and add their outputs as initializers
    folded_outputs = set(const_vals.keys()) - {
        init.name for init in model.graph.initializer
    }
    new_nodes = [
        n for n in model.graph.node if not any(o in folded_outputs for o in n.output)
    ]

    new_inits = list(model.graph.initializer)
    for name, val in const_vals.items():
        if name not in {init.name for init in model.graph.initializer}:
            new_inits.append(onnx.numpy_helper.from_array(val, name=name))

    # Rebuild graph in place
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
