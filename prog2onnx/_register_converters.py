from .common.operator_converters.dict_wrapper import dict_custom_converter
from .common.operator_converters.proglearn import prog_transformer_converter

from .common.shape_calculators.dict_wrapper import dict_shape_calculator
from .common.shape_calculators.proglearn import prog_transformer_shape_calculator

from .common.parsers._proglearn import progresive_parser

from .utils._dict_wrapper import DictWrapper

from ._supported_operators import proglearn_operator_name_map

from proglearn.progressive_learner import ClassificationProgressiveLearner
from proglearn.forest import LifelongClassificationForest
from skl2onnx import update_registered_converter

update_registered_converter(
    DictWrapper,
    proglearn_operator_name_map[DictWrapper],
    dict_shape_calculator,
    dict_custom_converter,
    overwrite=True,
)

update_registered_converter(
    ClassificationProgressiveLearner,
    proglearn_operator_name_map[ClassificationProgressiveLearner],
    prog_transformer_shape_calculator,
    prog_transformer_converter,
    parser=progresive_parser,
    overwrite=True,
)

update_registered_converter(
    LifelongClassificationForest,
    proglearn_operator_name_map[LifelongClassificationForest],
    prog_transformer_shape_calculator,
    prog_transformer_converter,
    parser=progresive_parser,
    overwrite=True,
)
