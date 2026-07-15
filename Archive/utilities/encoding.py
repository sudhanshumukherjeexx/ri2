import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


def PerformLabelEncoding(data, columns):
    encoder = LabelEncoder()
    for col in columns:
        data[col] = encoder.fit_transform(data[col])
    return data


def PerformOrdinalEncoding(data, columns):
    encoder = OrdinalEncoder()
    data[columns] = encoder.fit_transform(data[columns])
    return data


def PerformOneHotEncoding(data, columns):
    data = pd.get_dummies(data, columns=columns, dtype=float)
    return data
