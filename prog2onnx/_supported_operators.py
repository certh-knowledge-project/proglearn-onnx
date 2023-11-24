try:
    from proglearn.progressive_learner import ClassificationProgressiveLearner
except ImportError:
    ClassificationProgressiveLearner = None

try:
    from proglearn.forest import LifelongClassificationForest
except ImportError:
    LifelongClassificationForest = None

from .utils._dict_wrapper import DictWrapper


def build_proglearn_operator_name_map():
    res = {
        k: "ProgLearn" + k.__name__
        for k in [
            ClassificationProgressiveLearner,
            LifelongClassificationForest,
            DictWrapper,
        ]
        if k is not None
    }

    if None in res:
        del res[None]
    return res


def _get_proglearn_operator_name(model_type):
    """
    Get operator name of the input argument
    """
    if model_type not in proglearn_operator_name_map:
        # No proper operator name found, it means a local operator.
        alias = None
    else:
        alias = proglearn_operator_name_map[model_type]

    return alias


def get_model_alias(model_type):
    """
    Get alias model. Raise an exception if not found.
    """
    res = _get_proglearn_operator_name(model_type)
    if res is None:
        raise RuntimeError(
            "Unable to find alias for model '{}'. "
            "The converter is likely missing."
            "".format(model_type)
        )
    return res


# registered converters
proglearn_operator_name_map = build_proglearn_operator_name_map()
