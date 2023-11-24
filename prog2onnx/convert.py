import os.path as osp
import os
import onnx

from typing import Optional, Union, Callable
from proglearn.progressive_learner import ClassificationProgressiveLearner
from proglearn.forest import LifelongClassificationForest

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from onnx.checker import ValidationError
from numbers import Integral

from .utils.helpers.exceptions import ONNXConversionError
from .utils._constants import MIN_TARGET_OPSET, TARGET_AI_ONNX_ML, MAX_TARGET_OPSET


class Prog2ONNX:
    """
    A class to convert a progressive learning model to ONNX format.

    Parameters
    ----------
    model : Union[ClassificationProgressiveLearner, LifelongClassificationForest]
        The progressive learning model to be converted or the path to a saved model.

    Attributes
    ----------
    prog_model : Union[ClassificationProgressiveLearner, LifelongClassificationForest]
        The progressive learning model to be converted.
    onnx_model : onnx.ModelProto
        The ONNX representation of the progressive learning model.

    Methods
    -------
    to_onnx(task_id: int, save_path: Optional[Union[PathLike, str]] = None, target_opset: Optional[int] = 16, verbose: Optional[int] = 0, initial_types: Optional[str] = "float_input") -> onnx.ModelProto
        Converts the progressive learning model to ONNX format for the specified task ID.
    save(self, save_path: Union[os.PathLike, str], overwrite: bool = True) -> None
        Save the ONNX model to a specified path.
    validate(self, full: bool = False) -> None
        Validate the ONNX model
    """

    def __init__(
        self,
        model: Union[ClassificationProgressiveLearner, LifelongClassificationForest],
    ):
        self.prog_model = self._load_model(model)
        self.onnx_model = None

    def __repr__(self):
        return f"Prog2ONNX(prog_model={self.prog_model!r})"

    def __str__(self):
        return f"Prog2ONNX with prog_model of type {type(self.prog_model).__name__}"

    def _load_model(self, model):
        """
        Load/Import the model

        Notes
        -----
        `LifelongClassificationForest` inherits from `ClassificationProgressiveLearner`
        """
        if isinstance(model, ClassificationProgressiveLearner):
            return model
        else:
            raise TypeError(
                "`model` must be a %r instance"
                % ClassificationProgressiveLearner.__name__
            )

    def to_onnx(
        self,
        task_id: int,
        target_opset: Optional[int] = MIN_TARGET_OPSET,
        verbose: Optional[int] = 0,
        initial_types: Optional[str] = "float_input",
    ) -> onnx.ModelProto:
        """
        Converts the progressive learning model to ONNX format for the specified task ID.

        Parameters
        ----------
        task_id : int
            The task ID for which the progressive learning model should be converted.
        target_opset : Optional[int], default=16
            The target opset version for the ONNX conversion.
        verbose : Optional[int], default=0
            The verbosity level for the ONNX conversion.
        initial_types : Optional[str], default="float_input"
            The initial types for the ONNX conversion.

        Returns
        -------
        onnx.ModelProto:
            The ONNX representation of the model
        """

        if task_id not in self.prog_model.get_task_ids():
            raise ValueError("Invalid task_id: %d" % task_id)
        else:
            self.prog_model.task_id = task_id

        if isinstance(target_opset, (Integral, int)):
            if target_opset < MIN_TARGET_OPSET:
                raise ValueError(
                    f"Minimum working opset is {MIN_TARGET_OPSET} (> {target_opset})"
                )
            elif target_opset > MAX_TARGET_OPSET:
                raise ValueError(
                    f"Maximum working opset is {MAX_TARGET_OPSET} (< {target_opset})"
                )
        else:
            raise TypeError(
                f"`target_opset` must be of integer type, got {type(target_opset)}"
            )

        try:
            n_features = self.prog_model.task_id_to_X[0].shape[-1]
            self.onnx_model = convert_sklearn(
                self.prog_model,
                initial_types=[(initial_types, FloatTensorType([None, n_features]))],
                target_opset={"": target_opset, "ai.onnx.ml": TARGET_AI_ONNX_ML},
                verbose=verbose,
            )
        except Exception as e:
            raise ONNXConversionError(f"{e}")

        return self.onnx_model

    def _ensure_onnx(f: Callable) -> Callable:
        """
        Decorator to ensure that the ONNX model is loaded before calling a method.

        This decorator checks if the ONNX model has been loaded into the instance
        (i.e., `self.onnx_model` is not `None`). If the model is not loaded, it raises
        an exception. Otherwise, it proceeds to call the decorated method.

        Parameters
        ----------
        f (Callable):
            The method to be decorated.

        Returns
        -------
        Callable:
            The decorated method.
        """

        def wrapper(self, *args, **kwargs):
            if self.onnx_model is None:
                raise Exception("ONNX model not found! Please call `to_onnx` method.")
            return f(self, *args, **kwargs)

        return wrapper

    @_ensure_onnx
    def save(
        self, save_path: Union[os.PathLike, str], overwrite: Optional[bool] = True
    ) -> None:
        """
        Save the ONNX model to a specified path.

        This method saves the ONNX model to a specified path. If the path points to a directory,
        an IsADirectoryError is raised. If the directory does not exist, a NotADirectoryError is raised.
        If the file already exists and overwrite is False, a FileExistsError is raised.

        Parameters
        ----------
        save_path : Union[os.PathLike, str]
            The path where the ONNX model should be saved. This should be a os.PathLike instance or a valid string.
        overwrite : bool, optional
            If True, overwrite the existing file. If False, raise an error if the file exists. Default is True.

        Raises
        ------
        TypeError
            If save_path is not a os.PathLike instance or a valid string, or if overwrite is not a boolean value.
        IsADirectoryError
            If save_path points to a directory.
        NotADirectoryError
            If the directory of save_path does not exist.
        FileExistsError
            If the file already exists and overwrite is False.

        Returns
        -------
        None
        """
        if not isinstance(save_path, (os.PathLike, str)):
            raise TypeError("Path should be a os.PathLike instance or a valid string")
        if not isinstance(overwrite, bool):
            raise TypeError("Parameter `overwrite` must be a boolean value.")

        d, f = osp.splitext(
            osp.normpath(osp.normcase(osp.realpath(osp.expanduser(save_path))))
        )
        if osp.isdir(d) and not f:
            raise IsADirectoryError(
                "Cannot save file since path points to a directory."
            )
        elif not osp.isdir(d):
            fname = osp.basename(d) + ".onnx" if not f else osp.basename(d) + f
            if not osp.isdir(osp.dirname(d)):
                raise NotADirectoryError(f"'{osp.dirname(d)}' is not a directory.")

            save_path = osp.join(osp.dirname(d), fname)
            if (
                osp.exists(save_path) and osp.isfile(save_path) and overwrite
            ) or not osp.exists(save_path):
                with open(save_path, "wb") as outfile:
                    outfile.write(self.onnx_model.SerializeToString())
            else:
                raise FileExistsError(
                    f"Cannot overwrite existing file '{fname}' since `overwrite=False`"
                )

    @_ensure_onnx
    def validate(self, full: Optional[bool] = False) -> None:
        """
        Validate the ONNX model.

        Parameters
        ----------
        full : bool, optional
            If True, perform a full validation of the model.
            If False, perform a lighter validation.
            Default is False.

        Raises
        ------
        TypeError
            If `full` is not a boolean value.
        ValidationError
            If the ONNX model is invalid.

        Returns
        -------
        None
        """
        if not isinstance(full, bool):
            raise TypeError("Parameter `full` must be a boolean value.")

        try:
            onnx.checker.check_model(self.onnx_model, full_check=full)
        except ValidationError:
            raise ValidationError("ONNX model is invalid!")
