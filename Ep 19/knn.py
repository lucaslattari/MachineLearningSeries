from classification import ClassificationModel
from argumentparser import *

class KNN(ClassificationModel):
    def __init__(self, _args):
        self.args = _args

    def computeModel(XTrain, yTrain, _n_neighbors, power_parameter_minkowski_metric):
        from sklearn.neighbors import KNeighborsClassifier

        classifier = KNeighborsClassifier(n_neighbors = _n_neighbors, p = power_parameter_minkowski_metric)
        classifier.fit(XTrain, yTrain)

        return classifier

    def compute(self):
        import timeit
        start = timeit.default_timer()

        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(self.args, True)

        classifier = KNN.computeModel(XTrain, yTrain, self.args.n_neighbors, self.args.power_parameter_minkowski_metric)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        confusionMatrix = ClassificationModel.evaluateModel(yPred, yTest)

        if(self.args.print_accuracy):
            print(confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix))

        stop = timeit.default_timer()

        return confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix), stop - start

    def computeCrossValidation(self):
        from sklearn.model_selection import cross_validate

        X, y = ClassificationModel.preprocessDataCrossValidation(self.args, True)
        classifier = KNN.computeModel(X, y, self.args.n_neighbors, self.args.power_parameter_minkowski_metric)

        cv_results = cross_validate(classifier, X, y, cv=self.args.k_fold_cross_validation)

        if(self.args.print_accuracy):
            print(cv_results)

        return cv_results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.setBasicArguments()
    parser.setKNNArguments()
    args = parser.getArguments()

    model = KNN(args)

    if(args.cross_validation == False):
        model.compute()
    else:
        model.computeCrossValidation()
