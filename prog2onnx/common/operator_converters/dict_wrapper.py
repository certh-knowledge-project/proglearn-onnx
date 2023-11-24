import numpy as np

from skl2onnx.proto import onnx_proto
from skl2onnx.common._apply_operation import apply_identity


def dict_custom_converter(scope, operator, container):
    """
    Converts a custom dictionary operator to ONNX format.

    Parameters
    ----------
    scope : Scope
        The scope object for the current model.
    operator : Operator
        The operator object representing the custom dictionary operator.
    container : ModelComponentContainer
        The container object for the current model.

    Notes
    -----
    This function adds initializers and nodes to the container object to represent
    the custom dictionary operator. It uses the unique variable names generated by
    the scope object and the data from the operator object.
    """
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    dict_tensor = scope.get_unique_variable_name("dict_tensor")
    dict_values = np.asarray(list(op.data.values()), dtype=np.float32)
    container.add_initializer(
        dict_tensor,
        onnx_proto.TensorProto.FLOAT,
        dict_values.shape,
        dict_values.flatten(),
    )

    dict_keys = scope.get_unique_variable_name("dict_keys")
    container.add_initializer(
        dict_keys,
        onnx_proto.TensorProto.INT64,
        [len(op.data)],
        list(op.data.keys()),
    )

    equal_name = scope.get_unique_variable_name("equal")
    container.add_node(
        "Equal",
        [dict_keys, operator.inputs[0].full_name],
        equal_name,
        op_version=opv,
    )

    nonzero_name = scope.get_unique_variable_name("nonzero")
    container.add_node("NonZero", [equal_name], nonzero_name, op_version=opv)

    value_name = scope.get_unique_variable_name("value")
    container.add_node(
        "Gather", [dict_tensor, nonzero_name], value_name, op_version=opv
    )

    new_shape_value = scope.get_unique_variable_name("new_shape_value")
    container.add_initializer(new_shape_value, onnx_proto.TensorProto.INT64, [1], [-1])

    new_shape_1d = scope.get_unique_variable_name("new_shape_1d")
    container.add_node("Reshape", [value_name, new_shape_value], [new_shape_1d])

    apply_identity(scope, value_name, out[0].full_name, container)