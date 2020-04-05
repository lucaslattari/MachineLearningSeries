from classification import ClassificationModel
from argumentparser import *

class DecisionTree(ClassificationModel):
    def __init__(self, _args):
        self.args = _args

    def computeModel(XTrain, yTrain, _criterion):
        from sklearn.tree import DecisionTreeClassifier

        classifier = DecisionTreeClassifier(criterion = _criterion)
        classifier.fit(XTrain, yTrain)

        return classifier

    def compute(self):
        import timeit
        start = timeit.default_timer()

        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(self.args, False)

        classifier = DecisionTree.computeModel(XTrain, yTrain, self.args.criterion)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        confusionMatrix = ClassificationModel.getConfusionMatrix(yPred, yTest)
        rocCurve = ClassificationModel.getRocCurve(yPred, yTest)

        if(self.args.print_accuracy):
            print(confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix))

        stop = timeit.default_timer()

        return confusionMatrix, rocCurve, ClassificationModel.getAccuracy(confusionMatrix), stop - start, classifier

    def computeCrossValidation(self):
        from sklearn.model_selection import cross_validate

        X, y = ClassificationModel.preprocessDataCrossValidation(self.args, False)
        classifier = DecisionTree.computeModel(X, y, self.args.criterion)

        cv_results = cross_validate(classifier, X, y, cv=self.args.k_fold_cross_validation)

        if(self.args.print_accuracy):
            print(cv_results)

        return cv_results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.setBasicArguments()
    parser.setDecisionTreeArguments()
    args = parser.getArguments()

    model = DecisionTree(args)

    if(args.cross_validation == False):
        model.compute()
    else:
        model.computeCrossValidation()
