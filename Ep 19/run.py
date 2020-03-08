from logisticregression import LogisticRegression
from knn import KNN
from svm import SVM
from naivebayes import NaiveBayes
from decisiontree import DecisionTree
from randomforest import RandomForest

import numpy as np
import pandas as pd

from argumentparser import *

def returnTrueIfNoAlgorithmWasSelected(args):
    if(args.run_all == False):
        if(args.decision_tree == False):
            if(args.knn == False):
                if(args.logistic_regression == False):
                    if(args.naive_bayes == False):
                        if(args.random_forest == False):
                            return True
    return False

def computeDecisionTree(args, dict_algorithms):
    if(args.debug):
        print("Running decision tree...", end='')
    model = DecisionTree(args)
    dict_algorithms["decision_tree"] = model.compute()
    if(args.debug):
        print("ok!")

def computeDecisionTreeCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running decision tree...", end='')
    model = DecisionTree(args)
    dict_algorithms["decision_tree"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeRandomForest(args, dict_algorithms):
    if(args.debug):
        print("Running random forest...", end='')
    model = RandomForest(args)
    dict_algorithms["random_forest"] = model.compute()
    if(args.debug):
        print("ok!")

def computeRandomForestCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running random forest...", end='')
    model = RandomForest(args)
    dict_algorithms["random_forest"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeLogisticRegression(args, dict_algorithms):
    if(args.debug):
        print("Running logistic regression...", end='')
    model = LogisticRegression(args)
    dict_algorithms["logistic_regression"] = model.compute()
    if(args.debug):
        print("ok!")

def computeLogisticRegressionCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running logistic regression...", end='')
    model = LogisticRegression(args)
    dict_algorithms["logistic_regression"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeKNN(args, dict_algorithms):
    if(args.debug):
        print("Running knn...", end='')
    model = KNN(args)
    dict_algorithms["knn"] = model.compute()
    if(args.debug):
        print("ok!")

def computeKNNCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running knn...", end='')
    model = KNN(args)
    dict_algorithms["knn"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeNaiveBayes(args, dict_algorithms):
    if(args.debug):
        print("Running naive bayes...", end='')
    model = NaiveBayes(args)
    dict_algorithms["naive_bayes"] = model.compute()
    if(args.debug):
        print("ok!")

def computeNaiveBayesCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running naive bayes...", end='')
    model = NaiveBayes(args)
    dict_algorithms["naive_bayes"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeSVM(args, dict_algorithms):
    if(args.debug):
        print("Running svm...", end='')
    model = SVM(args)
    dict_algorithms["svm"] = model.compute()
    if(args.debug):
        print("ok!")

def computeSVMCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running svm...", end='')
    model = SVM(args)
    dict_algorithms["svm"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeAllAlgorithms(args, dict_algorithms):
    computeDecisionTree(args, dict_algorithms)
    computeRandomForest(args, dict_algorithms)
    computeLogisticRegression(args, dict_algorithms)
    computeNaiveBayes(args, dict_algorithms)
    computeKNN(args, dict_algorithms)
    computeSVM(args, dict_algorithms)

def computeAllAlgorithmsCrossValidation(args, dict_algorithms):
    computeDecisionTreeCrossValidation(args, dict_algorithms)
    computeRandomForestCrossValidation(args, dict_algorithms)
    computeLogisticRegressionCrossValidation(args, dict_algorithms)
    computeNaiveBayesCrossValidation(args, dict_algorithms)
    computeKNNCrossValidation(args, dict_algorithms)
    computeSVMCrossValidation(args, dict_algorithms)

def compute(args):
    if(args.clean_log):
        import os
        os.remove('MLCF.log')

    import logging
    logging.basicConfig(filename='MLCF.log', filemode='a', format='%(message)s', level=logging.INFO)

    dict_algorithms = {}
    if(returnTrueIfNoAlgorithmWasSelected(args)):
        print("ERRO! Selecione algum algoritmo para executar este programa. Use o argumento -h para imprimir uma tela com as opções possíveis")
        return
    elif(args.run_all):
        computeAllAlgorithms(args, dict_algorithms)
    else:
        if(args.decision_tree):
            computeDecisionTree(args, dict_algorithms)

        if(args.random_forest):
            computeRandomForest(args, dict_algorithms)

        if(args.logistic_regression):
            computeLogisticRegression(args, dict_algorithms)

        if(args.knn):
            computeKNN(args, dict_algorithms)

        if(args.naive_bayes):
            computeNaiveBayes(args, dict_algorithms)

        if(args.svm):
            computeSVM(args, dict_algorithms)

    if(args.sort_by_time):
        dict_algorithms_sorted = {k: v for k, v in sorted(dict_algorithms.items(), key=lambda item: item[1][2])}
    else:
        dict_algorithms_sorted = {k: v for k, v in sorted(dict_algorithms.items(), key=lambda item: item[1][1], reverse=True)}

    print(dict_algorithms_sorted)

    import time
    logging.info('--------------------------------------------------')
    logging.info(time.asctime(time.localtime()))
    logging.info("filename dataset: " + args.dataset)
    logging.info(dict_algorithms_sorted)
    logging.info(args)
    logging.info("")

def computeCrossValidation(args):
    if(args.clean_log):
        import os
        os.remove('MLCF.log')

    import logging
    logging.basicConfig(filename='MLCF.log', filemode='a', format='%(message)s', level=logging.INFO)

    dict_algorithms = {}
    if(returnTrueIfNoAlgorithmWasSelected(args)):
        print("ERRO! Selecione algum algoritmo para executar este programa. Use o argumento -h para imprimir uma tela com as opções possíveis")
        return
    elif(args.run_all):
        computeAllAlgorithmsCrossValidation(args, dict_algorithms)
    else:
        if(args.decision_tree):
            computeDecisionTreeCrossValidation(args, dict_algorithms)

        if(args.random_forest):
            computeRandomForestCrossValidation(args, dict_algorithms)

        if(args.logistic_regression):
            computeLogisticRegressionCrossValidation(args, dict_algorithms)

        if(args.knn):
            computeKNNCrossValidation(args, dict_algorithms)

        if(args.naive_bayes):
            computeNaiveBayesCrossValidation(args, dict_algorithms)

        if(args.svm):
            computeSVMCrossValidation(args, dict_algorithms)

    print(dict_algorithms)

    import time
    logging.info('--------------------------------------------------')
    logging.info(time.asctime(time.localtime()))
    logging.info("filename dataset: " + args.dataset)
    logging.info(dict_algorithms)
    logging.info(args)
    logging.info("")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.setBasicArguments()
    parser.setAllAlgorithmsArguments()
    args = parser.getArguments()
    if(args.run_all or args.random_forest):
        parser.setRandomForestArguments()
    if(args.run_all or args.logistic_regression):
        parser.setLogisticRegressionArguments()
    if(args.run_all or args.knn):
        parser.setKNNArguments()
    if(args.run_all or args.decision_tree):
        parser.setDecisionTreeArguments()
    if(args.run_all or args.svm):
        parser.setSVMArguments()
    args = parser.getArguments()

    if(args.cross_validation == False):
        compute(args)
    else:
        computeCrossValidation(args)
