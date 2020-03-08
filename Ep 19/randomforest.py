from classification import ClassificationModel
from argumentparser import *

class RandomForest(ClassificationModel):
    def __init__(self, _args):
        self.args = _args

    def computeModel(XTrain, yTrain, _n_estimators, _criterion):
        from sklearn.ensemble import RandomForestClassifier

        classifier = RandomForestClassifier(n_estimators = _n_estimators, criterion = _criterion)
        classifier.fit(XTrain, yTrain)

        return classifier

    def compute(self):
        import timeit
        start = timeit.default_timer()

        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(self.args, False)

        classifier = RandomForest.computeModel(XTrain, yTrain, self.args.n_estimators, self.args.criterion)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        confusionMatrix = ClassificationModel.evaluateModel(yPred, yTest)

        if(self.args.print_accuracy):
            print(confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix))

        stop = timeit.default_timer()

        return confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix), stop - start

    def computeCrossValidation(self):
        from sklearn.model_selection import cross_validate

        X, y = ClassificationModel.preprocessDataCrossValidation(self.args, False)
        classifier = RandomForest.computeModel(X, y, self.args.n_estimators, self.args.criterion)

        cv_results = cross_validate(classifier, X, y, cv=self.args.k_fold_cross_validation)

        if(self.args.print_accuracy):
            print(cv_results)

        return cv_results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.setBasicArguments()
    parser.setRandomForestArguments()
    args = parser.getArguments()

    model = RandomForest(args)

    if(args.cross_validation == False):
        model.compute()
    else:
        model.computeCrossValidation()
