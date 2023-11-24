def prog_transformer_shape_calculator(operator):
    """
    Calculates the output shape for a progressive transformer operator.

    Parameters
    ----------
    operator : Operator
        The operator object representing the progressive transformer operator.

    Notes
    -----
    This function sets the shape of the operator's outputs based on the size of the default decider in the operator's
    raw operator and the first dimension of the operator's inputs.
    """
    try:
        out_ft = list(operator.raw_operator.default_decider_kwargs.values())[0].size
    except IndexError:
        keys = list(operator.raw_operator.task_id_to_y.keys())
        out_ft = len(set(operator.raw_operator.task_id_to_y[keys[-1]]))

    N = operator.inputs[0].get_first_dimension()

    operator.outputs[0].type.shape = [N]
    operator.outputs[1].type.shape = [N, out_ft]
