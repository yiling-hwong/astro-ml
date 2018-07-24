__author__ = 'yi-linghwong'

import sys
import os
import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


class SGD():

    def train_test_split(self):

        #################
        # Split the dataset in training and test set:
        #################

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        print("Number of data point is " + str(len(y)))

        return x_train, x_test, y_train, y_test

    def stratified_shufflesplit(self):

        ####################
        # Stratified ShuffleSplit cross validation iterator
        # Provides train/test indices to split data in train test sets.
        # This cross-validation object is a merge of StratifiedKFold and ShuffleSplit,
        # which returns stratified randomized folds.
        # The folds are made by preserving the percentage of samples for each class.
        ####################


        sss = StratifiedShuffleSplit(y, 5, test_size=0.2, random_state=42)

        for train_index, test_index in sss:
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return x_train, x_test, y_train, y_test

    def stratified_kfolds(self):

        ##################
        # Stratified K-Folds cross validation iterator
        # Provides train/test indices to split data in train test sets.
        # This cross-validation object is a variation of KFold that returns stratified folds.
        # The folds are made by preserving the percentage of samples for each class.
        ##################

        skf = StratifiedKFold(y, n_folds=5)

        print(len(skf))
        print(skf)

        for train_index, test_index in skf:
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return x_train, x_test, y_train, y_test

    def train_classifier(self):

        # train the classifier

        print("Fitting data ...")
        clf = SGDClassifier(loss=_loss, penalty=_penalty, class_weight=_class_weight, alpha=_alpha,
                            n_iter=_n_iter, random_state=42).fit(x_train, y_train)


        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='f1_weighted')
        print ("Cross validation score: "+str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation

        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


        y_predicted = clf.predict(x_test)

        # print the mean accuracy on the given test data and labels

        print("Classifier score on test data is: %0.2f " % clf.score(x_test, y_test))

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf


    def train_classifier_use_feature_selection(self):


        #################
        # feature selection
        #################

        selector = SelectPercentile(score_func=_score_func, percentile=_percentile)

        print("Fitting data with feature selection ...")
        selector.fit(x_train, y_train)

        # get how many features are left after feature selection
        x_features = selector.transform(x_train)

        print("Shape of array after feature selection is " + str(x_features.shape))

        clf = SGDClassifier(loss=_loss, penalty=_penalty, alpha=_alpha,
                            n_iter=_n_iter, random_state=42).fit(x_features, y_train)

        # get the features which are selected and write to file

        feature_boolean = selector.get_support(indices=False)

        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, x_features, y_train, cv=10, scoring='f1_weighted')
        print ("Cross validation score: "+str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation

        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


        ####################
        # test clf on test data
        ####################


        # apply feature selection on test data too

        x_test_selector = selector.transform(x_test)

        print("Shape of array for test data after feature selection is " + str(x_test_selector.shape))

        y_predicted = clf.predict(x_test_selector)

        # print the mean accuracy on the given test data and labels

        print("Classifier score on test data is: %0.2f " % clf.score(x_test_selector, y_test))


        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf


    def use_pipeline_with_feature_selection(self):

        #####################
        # Build a classifier pipeline and carry out feature selection and grid search for best classif parameters
        #####################

        pipeline = Pipeline([
            ("selector", SelectPercentile()),
            ('clf', SGDClassifier(random_state=42))
        ])

        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'selector__score_func': (chi2, f_classif),
            'selector__percentile': (85, 95, 100),
            'clf__loss': ('hinge', 'log'),
            'clf__penalty': ('l2', 'l1', 'elasticnet'),
            'clf__n_iter': (5,10),
            'clf__alpha': (0.001, 0.0001, 0.0005),

        }

        #################
        # Exhaustive search over specified parameter values for an estimator, use cv to generate data to be used
        # implements the usual estimator API: when “fitting” it on a dataset all the possible combinations
        # of parameter values are evaluated and the best combination is retained.
        #################

        cv = StratifiedShuffleSplit(y_train, n_iter=5, test_size=0.2, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=cv, n_jobs=-1)
        clf_gs = grid_search.fit(x_train, y_train)

        ###############
        # print the cross-validated scores for the each parameters set explored by the grid search
        ###############

        best_parameters, score, _ = max(clf_gs.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        print("Score for gridsearch is %0.2f" % score)


        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        score_func = best_parameters['selector__score_func']
        percentile = best_parameters['selector__percentile']
        loss = best_parameters['clf__loss']
        penalty = best_parameters['clf__penalty']
        alpha = best_parameters['clf__alpha']


        #################
        # feature selection
        #################

        selector = SelectPercentile(score_func=score_func, percentile=percentile)

        combined_features = Pipeline([
            ("feat_select", selector)
        ])

        X_features = combined_features.fit_transform(x_train, y_train)
        X_test_features = combined_features.transform(x_test)

        print("Shape of train data after feature selection is " + str(X_features.shape))
        print("Shape of test data after feature selection is " + str(X_test_features.shape))

        # run classifier on selected features

        clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, random_state=42).fit(X_features, y_train)

        # get the features which are selected and write to file

        feature_boolean = selector.get_support(indices=False)

        f = open(path_to_store_feature_selection_boolean_file,'w')

        for fb in feature_boolean:
            f.write(str(fb)+'\n')

        f.close()


        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, X_features, y_train, cv=10, scoring='f1_weighted')
        print ("Cross validation score: "+str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation

        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


        ##################
        # run classifier on test data
        ##################


        y_predicted = clf.predict(X_test_features)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score on test data is: %0.2f " % clf.score(X_test_features,y_test))

        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)


        return clf



    def get_important_features(self,clf):

        # get coef_

        f = open(path_to_store_coefficient_file, 'w')

        coef = clf.coef_[0]

        for c in coef:
            f.write(str(c) + '\n')

        f.close()

        # get feature names

        feature_names = get_data_set()[2][:-1]

        f = open(path_to_store_list_of_feature_file, 'w')

        for fn in feature_names:
            f.write(str(fn) + '\n')
        f.close()

        print(len(coef))
        print(len(feature_names))

        #################
        # if feature selection was used, need to find out which are the features that are retained
        #################

        if len(coef) != len(feature_names):

            print()
            print("###### feature selection was used, getting retained features ######")

            lines = open(path_to_store_feature_selection_boolean_file).readlines()

            feature_boolean = []

            for line in lines:
                spline = line.replace('\n', '')
                feature_boolean.append(spline)

            if len(feature_boolean) == len(feature_names):

                selected_features = zip(feature_names, feature_boolean)

                feature_names = []

                for sf in selected_features:
                    if sf[1] == 'True':
                        feature_names.append(sf[0])

                print("Length of retained features is " + str(len(feature_names)))

            else:
                print("length not equal, exiting...")
                sys.exit()

        # sort feature importance

        coef_list = []
        for c in coef:
            coef_list.append(c)

        if len(coef_list) == len(feature_names):

            feat_list = list(zip(coef_list, feature_names))

            feat_list.sort()

        else:
            print("Length of coef and feature list not equal, exiting...")
            sys.exit()

        f = open(path_to_store_feature_and_coef_file, 'w')

        # f.write(str(feat_list))

        for fl in feat_list:
            f.write(str(fl) + '\n')
        f.close()

        her = []
        ler = []

        for fl in feat_list:

            if fl[0] < 0:
                her.append(['HER', fl[1], str(fl[0])])

            if fl[0] > 0:
                ler.append(['LER', fl[1], str(fl[0])])

        f = open(path_to_store_important_features_by_class_file, 'w')

        for feat in her[:100]:
            f.write(','.join(feat) + '\n')

        for feat in reversed(ler[-100:]):
            f.write(','.join(feat) + '\n')

        f.close()


    def plot_feature_selection(self):

        # Plot classifier performance vs. percentila of top features used

        transform = SelectPercentile(score_func=_score_func)

        clf = Pipeline([('anova', transform), ('clf', SGDClassifier(loss=_loss, penalty=_penalty,
                                                                    alpha=_alpha, random_state=42))])

        ###############################################################################
        # Plot the cross-validation score as a function of percentile of features
        score_means = list()
        score_stds = list()
        percentiles = (10, 20, 30, 40, 60, 80, 85, 95, 100)

        for percentile in percentiles:
            clf.set_params(anova__percentile=percentile)
            # Compute cross-validation score using all CPUs
            this_scores = cross_validation.cross_val_score(clf, x_train, y_train, n_jobs=-1)
            score_means.append(this_scores.mean())
            score_stds.append(this_scores.std())

        plt.errorbar(percentiles, score_means, np.array(score_stds))

        plt.title('Performance of the SVM-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')

        plt.axis('tight')
        plt.show()


###############
# Set paths
###############

path_to_labelled_file = 'path_to_labelled_file'

path_to_store_coefficient_file = 'path_to_store_coefficient_file'
path_to_store_feature_selection_boolean_file = 'path_to_store_feature_selection_boolean_file'
path_to_store_list_of_feature_file = 'path_to_store_list_of_feature_file'
path_to_store_feature_and_coef_file = 'path_to_store_feature_and_coef_file'
path_to_store_important_features_by_class_file = 'path_to_store_important_features_by_class_file'

# hyperparameters (for non-grid-search)

_loss = 'log' # 'hinge' gives linear SVM; 'log' gives logistic regression
_penalty = 'l2'
_n_iter = 5
_alpha = 0.001
_score_func = chi2
_percentile = 85
_class_weight = "balanced"


def get_data_set():

    #############
    # Get dataset
    #############


    lines = open(path_to_labelled_file, 'r').readlines()

    for line in lines[:1]:
        spline = line.rstrip('\n').split(',')

        column_names = spline

    print(column_names)

    dataset = pd.read_csv(path_to_labelled_file, names=column_names)
    print(dataset.shape)

    X = dataset.ix[1:, :-1]
    y = dataset['label'][1:]

    return X,y,column_names



if __name__ == '__main__':

    X = get_data_set()[0]
    y = get_data_set()[1]

    sgd = SGD()

    ###################
    # select (uncomment) one of the methods to split data using Cross Validation
    ###################

    x_train,x_test,y_train,y_test = sgd.train_test_split()
    #x_train,x_test,y_train,y_test = sgd.stratified_shufflesplit()
    #x_train,x_test,y_train,y_test = sgd.stratified_kfolds()

    ##################
    # run SGD Classifier
    ##################

    clf = sgd.train_classifier()


    ###################
    # run SGD Classifier and use feature selection
    ###################

    #clf = sgd.train_classifier_use_feature_selection()


    ###################
    # use pipeline and use feature selection
    ###################

    #clf = sgd.use_pipeline_with_feature_selection()

    ###################
    # Get feature importance
    ###################

    sgd.get_important_features(clf)


    ##################
    # Plot feature selection
    ##################

    #sgd.plot_feature_selection()