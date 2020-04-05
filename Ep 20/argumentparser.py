import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def setBasicArguments(self):
        self.parser.add_argument('dataset', help="filename of dataset (csv file format)")
        self.parser.add_argument('-deli', dest='delimiter', default=',', required=False, type=str, help="delimiter of each column of csv")
        self.parser.add_argument('-missing', default = 2, dest='fill_missing_data_columns', required=False, type=str, help="use fill missing data? (if yes, enter column numbers separated by commas)")
        self.parser.add_argument('-one_hot', default = '0,1', dest='one_hot_encoding_columns', required=False, type=str, help="use one hot encoding? (if yes, enter column numbers separated by commas)")
        self.parser.add_argument('-test_size', dest='test_size', default=0.2, type=float, help="size of test set compared to train test")
        self.parser.add_argument('-print', dest='print_accuracy', action='store_true', help="print accuracy of method(s)")
        self.parser.add_argument('--version', action='version', version='%(prog)s 0.1')

        self.parser.add_argument('--cv', dest='cross_validation', action='store_true', help="activates cross validation.")
        self.parser.add_argument('-kf', dest='k_fold_cross_validation', default = 3, type=int, help="Determines the cross-validation splitting strategy (size of train and test partitions)")

    def setRandomForestArguments(self):
        self.parser.add_argument('-ne', dest='n_estimators', default=100, type=int, help="number of trees in the forest.")
        tempArgs = self.parser.parse_args()
        if(hasattr(tempArgs, 'criterion') == False):
            self.parser.add_argument('-c', dest='criterion', default='entropy', type=str, help="function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.")

    def setLogisticRegressionArguments(self):
        self.parser.add_argument('-sol', dest='solver', default = 'lbfgs', help="Algorithm to use in the optimization problem. For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones. For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes. ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty; ‘liblinear’ and ‘saga’ also handle L1 penalty; ‘saga’ also support ‘elasticnet’ penalty; ‘liblinear’ does not support setting penalty='none'. Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale.")

    def setKNNArguments(self):
        self.parser.add_argument('-n', dest='n_neighbors', default=5, type=int, help="number of neighbors to use by default for kneighbors queries.")
        self.parser.add_argument('-p', dest='power_parameter_minkowski_metric', default=2, type=int, help="the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.")

    def setDecisionTreeArguments(self):
        tempArgs = self.parser.parse_args()
        if(hasattr(tempArgs, 'criterion') == False):
            self.parser.add_argument('-c', dest='criterion', default='entropy', type=str, help="function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.")

    def setSVMArguments(self):
        self.parser.add_argument('-k', dest='kernel', default = 'linear', help="Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used.")

    def setAllAlgorithmsArguments(self):
        self.parser.add_argument('-RF', dest='random_forest', action="store_true", required=False, help="use random forest?")
        self.parser.add_argument('-DT', dest='decision_tree', action="store_true", required=False, help="use decision tree?")
        self.parser.add_argument('-LR', dest='logistic_regression', action="store_true", required=False, help="use logistic regression?")
        self.parser.add_argument('-KNN', dest='knn', action="store_true", required=False, help="use knn?")
        self.parser.add_argument('-NB', dest='naive_bayes', action="store_true", required=False, help="use naive bayes?")
        self.parser.add_argument('-SVM', dest='svm', action="store_true", required=False, help="use svm?")
        self.parser.add_argument('-ALL', dest='run_all', action="store_true", required=False, help="use all algorithms?")
        self.parser.add_argument('-time', dest='sort_by_time', action="store_true", required=False, help="sort algorithms by time, if more than one is being computed")
        self.parser.add_argument('--debug', action="store_true", required=False, help="print debug")
        self.parser.add_argument('--cl', dest='clean_log', action="store_true", required=False, help="erase log file")

    def getArguments(self):
        return self.parser.parse_args()
