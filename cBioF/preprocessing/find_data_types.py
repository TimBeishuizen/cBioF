# -*- coding: utf-8 -*-
"""Generic finding out the column types of the matrix"""

# Author: T.P.A. Beishuizen <tim.beishuizen@gmail.com>

def column_types_dataset(X, categorical=True):
    """ Find the types of the dataset

    :param X: The dtype of the first value
    :param categorical: Whether categorical is True or False. Default: True
    :return: A list with the column types
    """

    column_types = []

    def dtype_is_categorical(dtype, categorical=True):
        if categorical:
            return not("int" in str(dtype) or "float" in str(dtype))
        else:
            return ("int" in str(dtype) or "float" in str(dtype))

    for i in range(X.shape[1]):
        column_types.append(dtype_is_categorical(type(X[0, i]), categorical))

    return column_types
