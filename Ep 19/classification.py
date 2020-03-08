import preprocessing as pre
import numpy as np

class ClassificationModel:
    def getAccuracy(confusionMatrix):
        accuracy = (confusionMatrix[0][0] + confusionMatrix[1][1]) / (confusionMatrix[0][0] + confusionMatrix[1][0] + confusionMatrix[0][1] + confusionMatrix[1][1])
        return accuracy * 100

    def predictModel(classifier, X):
        return classifier.predict(X)

    def evaluateModel(yPred, yTest):
        from sklearn.metrics import confusion_matrix
        confusionMatrix = confusion_matrix(yTest, yPred)

        return confusionMatrix

    def preprocessData(args, use_scaling):
        X, y, csv = pre.loadDataset(args.dataset, args.delimiter)

        if(args.fill_missing_data_columns is not None):
            columns = args.fill_missing_data_columns.split(',')
            columns = [ int(x) for x in columns ]

            offset = 0
            for n in columns:
                X = pre.fillMissingData(X, n + offset)
                offset += n

        if(args.one_hot_encoding_columns is not None):
            columns = args.one_hot_encoding_columns.split(',')
            columns = [ int(x) for x in columns ]

            offset = 0
            for n in columns:
                X, o = pre.computeCategorization(X, n + offset)
                offset += o - 1

        XTrain, XTest, yTrain, yTest = pre.splitTrainTestSets(X, y, args.test_size)

        if(use_scaling == True):
            XTrain = pre.computeScaling(XTrain)
            XTest = pre.computeScaling(XTest)

        if(len(XTrain) == 2):
            XTrain = XTrain[0]
        if(len(XTest) == 2):
            XTest = XTest[0]

        return XTrain, XTest, yTrain, yTest

    def preprocessDataCrossValidation(args, use_scaling):
        X, y, csv = pre.loadDataset(args.dataset, args.delimiter)

        if(args.fill_missing_data_columns is not None):
            columns = args.fill_missing_data_columns.split(',')
            columns = [ int(x) for x in columns ]

            offset = 0
            for n in columns:
                X = pre.fillMissingData(X, n + offset)
                offset += n

        if(args.one_hot_encoding_columns is not None):
            columns = args.one_hot_encoding_columns.split(',')
            columns = [ int(x) for x in columns ]

            offset = 0
            for n in columns:
                X, o = pre.computeCategorization(X, n + offset)
                offset += o - 1

        if(use_scaling == True):
            X = pre.computeScaling(X)

        if(len(X) == 2):
            X = X[0]

        return X, y
