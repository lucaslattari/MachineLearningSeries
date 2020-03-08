from classification import ClassificationModel
from argumentparser import *

class NaiveBayes(ClassificationModel):
    def __init__(self, _args):
        self.args = _args

    def computeModel(XTrain, yTrain):
        from sklearn.naive_bayes import GaussianNB

        classifier = GaussianNB()
        classifier.fit(XTrain, yTrain)

        return classifier

    def compute(self):
        import timeit
        start = timeit.default_timer()

        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(self.args, True)

        classifier = NaiveBayes.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        confusionMatrix = ClassificationModel.evaluateModel(yPred, yTest)

        if(self.args.print_accuracy):
            print(confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix))

        stop = timeit.default_timer()

        return confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix), stop - start

    def computeCrossValidation(self):
        from sklearn.model_selection import cross_validate

        X, y = ClassificationModel.preprocessDataCrossValidation(self.args, True)
        classifier = NaiveBayes.computeModel(X, y)

        cv_results = cross_validate(classifier, X, y, cv=self.args.k_fold_cross_validation)

        if(self.args.print_accuracy):
            print(cv_results)

        return cv_results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.setBasicArguments()
    args = parser.getArguments()

    model = NaiveBayes(args)

    if(args.cross_validation == False):
        model.compute()
    else:
        model.computeCrossValidation()
