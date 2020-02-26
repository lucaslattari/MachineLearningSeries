from classification import ClassificationModel

class RandomForest(ClassificationModel):
    def computeModel(XTrain, yTrain):
        from sklearn.ensemble import RandomForestClassifier

        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
        classifier.fit(XTrain, yTrain)

        return classifier

    def computeExample(filename):
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(filename, False)

        classifier = RandomForest.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest, True)
        return ClassificationModel.evaluateModel(yPred, yTest)

if __name__ == "__main__":
    print(RandomForest.computeExample("titanic.csv"))
