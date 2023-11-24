import numpy as np
import skl2onnx.proto as onnx_proto

from skl2onnx.common.utils import check_input_and_output_types
from skl2onnx.common.data_types import FloatTensorType, guess_numpy_type
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from skl2onnx.algebra.onnx_ops import (
    OnnxCast,
    OnnxSqueeze,
    OnnxSlice,
    OnnxIdentity,
    OnnxDiv,
    OnnxArgMax,
    OnnxAdd,
)

from ...utils.helpers.dict_utils import fill_missing_indices
from ...utils._dict_wrapper import DictWrapper


def prog_transformer_converter(scope, operator, container):
    """
    Converts a progressive transformer operator to ONNX format.

    Parameters
    ----------
    scope : Scope
        The scope object for the current model.
    operator : Operator
        The operator object representing the progressive transformer operator.
    container : ModelComponentContainer
        The container object for the current model.

    Notes
    -----
    This function checks the input and output types of the operator, retrieves the task ID from the raw operator,
    and determines whether the default voter class is a tree voter or a KNN voter. It then creates an ONNX subgraph
    for each estimator in the raw operator, adds them to the total probability, and adds nodes to the container to
    calculate the final probability and anomaly score. If the default voter class is not recognized, it raises a
    ValueError.
    """
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)

    if hasattr(op, "task_id"):
        task_id = op.task_id
    else:
        raise AttributeError(
            f"`task_id` attribute not found in `{op.__class__.__name__}` object"
        )

    tree_voter = True
    voters = np.asarray(
        list(op.task_id_to_transformer_id_to_voters[task_id].values())
    ).flatten()
    n_fitted = list(op.task_id_to_y.values())[-1].size
    if op.default_voter_class.__name__ == "TreeClassificationVoter":
        max_keys = max(list(map(lambda x: max(x.leaf_to_posterior_.keys()), voters)))
        posterior_length = len(list(voters[0].leaf_to_posterior_.values())[0])
        scale_ = (n_fitted // max_keys) + 1
    elif op.default_voter_class.__name__ == "KNNClassificationVoter":
        tree_voter ^= tree_voter
    else:
        raise ValueError("Unknown voter class: %r" % op.default_voter_class.__name__)

    n_classes = np.unique(op.task_id_to_y[task_id]).size
    total_prob = np.asarray([[n_classes * [0.0]]], dtype=dtype)
    transformers = np.asarray(
        list(op.transformer_id_to_transformers.values())
    ).flatten()
    estimators = transformers.copy()
    for i, estimator in enumerate(transformers):
        if tree_voter:
            est = OnnxSubEstimator(
                estimator.transformer_,
                X,
                op_version=opv,
                options={"decision_leaf": True, "zipmap": False},
            )
            leaf = OnnxIdentity(est[2], op_version=opv)
            posterior = OnnxSubEstimator(
                DictWrapper(
                    fill_missing_indices(
                        voters[i].leaf_to_posterior_,
                        max_keys,
                        posterior_len=posterior_length,
                        scale=scale_,
                    )
                ),
                leaf,
                op_version=opv,
            )
            posterior_val = OnnxIdentity(posterior, op_version=opv)
        else:
            raise NotImplementedError("`KNNClassificationVoter` not implemented.")

        total_prob = OnnxAdd(posterior_val, total_prob)

    if tree_voter:
        total_prob_final = OnnxDiv(
            total_prob,
            np.asarray([len(estimators)], dtype=dtype),
            op_version=opv,
        )
        total_prob_sliced = OnnxSlice(
            total_prob_final,
            np.array([1]),
            np.array([2]),
            np.array([0]),
            op_version=opv,
        )
        total_prob_final_sq = OnnxSqueeze(
            total_prob_sliced,
            op_version=opv,
            output_names=out[1],
        )
        anomaly = OnnxArgMax(total_prob_sliced, axis=2)
    else:
        pass

    total_prob_final_sq.add_to(scope, container)
    anomaly_flat = OnnxCast(
        OnnxSqueeze(anomaly),
        op_version=opv,
        to=onnx_proto.TensorProto.UINT8,
        output_names=out[:1],
    )
    anomaly_flat.add_to(scope, container)
