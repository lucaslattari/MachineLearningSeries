from classification import ClassificationModel
from argumentparser import *

class LogisticRegression(ClassificationModel):
    def __init__(self, _args):
        self.args = _args

    def computeModel(XTrain, yTrain, _solver):
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(solver=_solver)
        classifier.fit(XTrain, yTrain)

        return classifier

    def compute(self):
        import timeit
        start = timeit.default_timer()

        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(self.args, True)

        classifier = LogisticRegression.computeModel(XTrain, yTrain, self.args.solver)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        confusionMatrix = ClassificationModel.evaluateModel(yPred, yTest)

        if(self.args.print_accuracy):
            print(confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix))

        stop = timeit.default_timer()

        return confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix), stop - start

    def computeCrossValidation(self):
        from sklearn.model_selection import cross_validate

        X, y = ClassificationModel.preprocessDataCrossValidation(self.args, True)
        classifier = LogisticRegression.computeModel(X, y, self.args.solver)

        cv_results = cross_validate(classifier, X, y, cv=self.args.k_fold_cross_validation)

        if(self.args.print_accuracy):
            print(cv_results)

        return cv_results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.setBasicArguments()
    parser.setLogisticRegressionArguments()
    args = parser.getArguments()

    model = LogisticRegression(args)

    if(args.cross_validation == False):
        model.compute()
    else:
        model.computeCrossValidation()
