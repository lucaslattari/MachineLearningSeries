import numpy as np
import pandas as pd

def loadDataset(filename, deli):
    baseDeDados = pd.read_csv(filename, delimiter=deli)
    X = baseDeDados.iloc[:,:-1].values
    y = baseDeDados.iloc[:,-1].values

    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    y = labelencoder_X.fit_transform(y)

    return X, y, baseDeDados

def fillMissingData(X, column):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X[:,column:column + 1] = imputer.fit_transform(X[:,column:column + 1])
    return X

def computeCategorization(X, column):
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:, column] = labelencoder_X.fit_transform(X[:, column])

    #one hot encoding
    D = pd.get_dummies(X[: , column]).values

    X = np.delete(X, column, 1)
    col = 0
    for ii in range(0, D.shape[1]):
        X = np.insert(X, column, D[:,ii], axis=1)
        col += 1

    return X, col

def splitTrainTestSets(X, y, testSize):
    from sklearn.model_selection import train_test_split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = testSize)
    return XTrain, XTest, yTrain, yTest

def computeScaling(X):
    from sklearn.preprocessing import StandardScaler
    scaleobj = StandardScaler()
    X = scaleobj.fit_transform(X.astype(float))

    return X, scaleobj
