"""
Main entry point to the converter from the *prog-learn* to *onnx*.
"""

__version__ = "0.0.1"
__author__ = ""  # TODO
__producer__ = "prog2onnx"
__producer_version__ = __version__
__domain__ = "ai.onnx"
__model_version__ = 0
__max_supported_opset__ = 18

from . import _register_converters
from .convert import Prog2ONNX

__all__ = ["Prog2ONNX"]
