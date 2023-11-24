import numpy as np
from sklearn.base import BaseEstimator


class DictWrapper(BaseEstimator):
    """
    A wrapper for dictionary data that provides transformation and fitting capabilities.

    Parameters
    ----------
    data : dict
        The dictionary data to be wrapped.

    Attributes
    ----------
    data : dict
        The dictionary data to be wrapped.
    """

    def __init__(self, data):
        self.data = data

    def transform(self, X):
        """
        Transforms the input by returning the corresponding value from the dictionary.

        Parameters
        ----------
        X : str
            The key to look up in the dictionary.

        Returns
        -------
        Any
            The value corresponding to the key in the dictionary.
        """
        return self.data.get(X)

    def fit(self, X, y=None):
        """
        A placeholder fit method. Does nothing.

        Parameters
        ----------
        X : array-like
            The input samples.
        y : array-like, default=None
            The target values. Ignored.
        """
        pass

    def __len__(self):
        """
        Returns the length of the first value in the dictionary.

        Returns
        -------
        int
            The length of the first value in the dictionary.
        """
        return len(list(self.data.values())[0])
