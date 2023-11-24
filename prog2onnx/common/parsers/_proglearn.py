from skl2onnx.common.data_types import UInt8TensorType
from skl2onnx import get_model_alias


def progresive_parser(scope, model, inputs, custom_parsers=None):
    """
    Parses a progressive learning model to a format compatible with ONNX.

    Parameters
    ----------
    scope : Scope
        The scope object for the current model.
    model : object
        The progressive learning model to be parsed.
    inputs : list
        The list of input variables for the model.
    custom_parsers : dict, default=None
        A dictionary of custom parsers to use for specific types of models.

    Returns
    -------
    list
        The list of output variables for the model.

    Notes
    -----
    This function declares a local operator for the model and adds the input variables to it. It then declares two
    local variables for the outputs and adds them to the operator. The output variables are logged and returned.
    """
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)

    # inputs
    this_operator.inputs.append(inputs[0])
    cls_type = inputs[0].type.__class__

    # outputs
    val_y1 = scope.declare_local_variable("class", UInt8TensorType())
    val_y2 = scope.declare_local_variable("probability", cls_type())

    this_operator.outputs.append(val_y1)
    this_operator.outputs.append(val_y2)

    return this_operator.outputs
