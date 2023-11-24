class ONNXConversionError(Exception):
    """
    Custom exception class for ONNX conversion failures.

    Attributes
    ----------
    message : str
        The error message to be displayed when the exception is raised.
    """

    __module__ = Exception.__module__

    def __init__(self, message: str):
        """
        Initialize the ONNXConversionFail exception.

        Parameters
        ----------
        message : str
            The error message to be displayed when the exception is raised.
        """
        self.message = message
