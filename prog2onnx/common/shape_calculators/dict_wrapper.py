from skl2onnx.common.data_types import FloatTensorType


def dict_shape_calculator(operator):
    """
    Calculates the output shape for a custom dictionary operator.

    Parameters
    ----------
    operator : Operator
        The operator object representing the custom dictionary operator.

    Notes
    -----
    This function sets the type of the operator's output to a float tensor with a shape
    determined by the length of the operator's data.
    """
    op = operator.raw_operator
    operator.outputs[0].type = FloatTensorType([len(op)])
