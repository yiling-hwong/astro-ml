__author__ = 'yi-linghwong'

import sys
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


class ExtraTree():

    def train_test_split(self):

        #################
        # Split the dataset in training and test set:
        #################

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        print ()
        print ("----------------------------")
        print("Number of data point is " + str(len(y)))

        return x_train, x_test, y_train, y_test

    def stratified_shufflesplit(self):

        ####################
        # Stratified ShuffleSplit cross validation iterator
        # Provides train/test indices to split data in train test sets.
        # This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds.
        # The folds are made by preserving the percentage of samples for each class.
        ####################


        sss = StratifiedShuffleSplit(y, 5, test_size=0.2, random_state=42)

        for train_index, test_index in sss:
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
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
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return x_train, x_test, y_train, y_test


    def train_classifier(self):


        print ("Fitting data ...")
        clf = ExtraTreesClassifier(n_estimators=_n_estimators, class_weight=_class_weight, criterion=_criterion, max_depth=_max_depth, min_samples_split=_min_samples_split).fit(x_train, y_train)

        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='f1_weighted')
        print ("Cross validation score: "+str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation

        print("Cross validation accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


        ##################
        # run classifier on test data
        ##################


        y_predicted = clf.predict(x_test)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score on test data is: %0.5f " % clf.score(x_test,y_test))

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf


    def use_pipeline(self):

        #####################
        # Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([

            ('clf', ExtraTreesClassifier()),
        ])

        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {

            'clf__n_estimators': (10, 50, 100),
            'clf__criterion': ("gini", "entropy"),
            'clf__max_depth': (None,2,4),
            'clf__min_samples_split': (2,4,6),
        }

        #################
        # Exhaustive search over specified parameter values for an estimator, use cv to generate data to be used
        # implements the usual estimator API: when “fitting” it on a dataset all the possible combinations of parameter values are evaluated and the best combination is retained.
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

        # y_predicted = clf_gs.predict(docs_test)


        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        n_estimators = best_parameters['clf__n_estimators']
        criterion = best_parameters['clf__criterion']
        max_depth = best_parameters['clf__max_depth']
        min_samples_split = best_parameters['clf__min_samples_split']


        # train the classifier

        print("Fitting data with best parameters ...")
        clf = ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split).fit(x_train, y_train)

        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='f1_weighted')
        print("Cross validation score: " + str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation

        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        ##################
        # run classifier on test data
        ##################

        y_predicted = clf.predict(x_test)

        # print the mean accuracy on the given test data and labels

        print("Classifier score on test data is: %0.2f " % clf.score(x_test, y_test))

        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf


    def use_pipeline_with_fs(self):

        #####################
        # Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([
            ("selector", SelectPercentile()),
            ('clf', ExtraTreesClassifier()),
        ])

        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {

            'clf__n_estimators': (10, 50, 100),
            'clf__criterion': ("gini", "entropy"),
            'clf__max_depth': (None, 2, 4),
            'clf__min_samples_split': (2, 4, 6),
            'selector__score_func': (chi2, f_classif),
            'selector__percentile': (85, 95, 100),
        }

        #################
        # Exhaustive search over specified parameter values for an estimator, use cv to generate data to be used
        # implements the usual estimator API: when “fitting” it on a dataset all the possible combinations of parameter values are evaluated and the best combination is retained.
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

        # y_predicted = clf_gs.predict(docs_test)

        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        n_estimators = best_parameters['clf__n_estimators']
        criterion = best_parameters['clf__criterion']
        max_depth = best_parameters['clf__max_depth']
        min_samples_split = best_parameters['clf__min_samples_split']
        score_func = best_parameters['selector__score_func']
        percentile = best_parameters['selector__percentile']


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

        clf = ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split).fit(X_features, y_train)

        # get the features which are selected and write to file

        feature_boolean = selector.get_support(indices=False)

        f = open(path_to_store_feature_selection_boolean_file, 'w')

        for fb in feature_boolean:
            f.write(str(fb) + '\n')

        f.close()

        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, X_features, y_train, cv=10, scoring='f1_weighted')
        print("Cross validation score: " + str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation

        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        #################
        # run classifier on test data
        #################

        y_predicted = clf.predict(X_test_features)

        # print the mean accuracy on the given test data and labels

        print("Classifier score on test data is: %0.2f " % clf.score(X_test_features, y_test))

        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf


    def get_important_features(self, clf):

        # get vocabulary
        feature_names = get_data_set()[2][:-1]

        f = open(path_to_store_vocabulary_file, 'w')

        for fn in feature_names:
            f.write(str(fn) + '\n')
        f.close()

        # get most important features
        feat_importance = clf.feature_importances_
        feat_imp = []

        for fi in feat_importance:
            feat_imp.append(str(fi))

        print(len(feature_names))
        print(len(feat_imp))

        if len(feature_names) == len(feat_imp):

            zipped = zip(feature_names, feat_imp)

            feature_and_scores = []
            feature_score_dict = {}

            for z in zipped:
                z = list(z)
                feature_and_scores.append(z)
                feature_score_dict[z[0]] = z[1]

            feature_and_scores.sort(key=lambda x: float(x[1]),
                                    reverse=True)  # sort list by most important feature first

            print(feature_and_scores)
            print(feature_score_dict)

            feat_scores = []

            for fs in feature_and_scores:
                fs[1] = str(fs[1])
                feat_scores.append(fs)

            f = open(path_to_store_feature_importance_file, 'w')

            for fs in feat_scores:
                f.write(','.join(fs) + '\n')

            f.close()

        #################
        # if feature selection was used, need to find out which are the features that are retained
        #################

        if len(feature_names) != len(feat_imp):

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
                print()

            else:
                print("length not equal, exiting...")
                sys.exit()

        f = open(path_to_store_complete_feature_importance_file, 'w')

        for fea in feat_imp:
            f.write(str(fea) + '\n')
        f.close()

        if len(feature_names) == len(feat_imp):

            zipped = zip(feat_imp, feature_names)
            feat_list = []

            for z in zipped:
                z = list(z)
                feat_list.append(z)

            feat_list.sort(reverse=True)

        else:
            print("Length of coef and feature list not equal, exiting...")
            sys.exit()

        f = open(path_to_store_top_important_features_file, 'w')

        for fl in feat_list[:100]:
            f.write('\t'.join(fl) + '\n')

        f.close()

        ##################
        # get feature importance by class
        ##################

        lines = open(path_to_labelled_file, 'r').readlines()

        her = []
        ler = []

        for line in lines[1:]:
            spline = line.rstrip('\n').split(',')

            if spline[-1] == 'nonprofit':

                temp = []

                for s in spline[:-1]:
                    t = float(s)
                    temp.append(t)

                her.append(temp)

            if spline[-1] == 'space':

                temp = []

                for s in spline[:-1]:
                    t = float(s)
                    temp.append(t)

                ler.append(temp)

        her_mean = np.mean(her,axis=0)
        ler_mean = np.mean(ler,axis=0)

        #print (len(her_mean),len(ler_mean))

        print ()
        #print ("HER mean:",her_mean)
        #print ("LER mean:",ler_mean)

        if len(her_mean) == len(ler_mean) == len(feature_names):

            zipped_her = zip(her_mean,feature_names)
            zipped_ler = zip(ler_mean,feature_names)

            temp_her = []
            temp_ler = []

            for z in zipped_her:
                z = list(z)
                temp_her.append(z)

            for z in zipped_ler:
                z = list(z)
                temp_ler.append(z)

            her_feat_imp_scores = []
            ler_feat_imp_scores = []

            for th in temp_her:

                mean_value = th[0]
                feat_imp_value = float(feature_score_dict[th[1]])

                feat_score = mean_value*feat_imp_value

                her_feat_imp_scores.append([th[1],feat_score])

            for tl in temp_ler:

                mean_value = tl[0]
                feat_imp_value = float(feature_score_dict[tl[1]])

                feat_score = mean_value*feat_imp_value

                ler_feat_imp_scores.append([tl[1],feat_score])

            print (len(her_feat_imp_scores),len(ler_feat_imp_scores))

            her_feat_imp_scores.sort(reverse=True, key=lambda x: x[1])
            ler_feat_imp_scores.sort(reverse=True, key=lambda x: x[1])

            her_final = []
            ler_final = []

            for h in her_feat_imp_scores:
                h[1] = str(h[1])
                her_final.append(['HER',h[0],h[1]])

            for l in ler_feat_imp_scores:
                l[1] = str(l[1])
                ler_final.append(['LER',l[0],l[1]])

            her_and_ler = her_final + ler_final

            # feat_by_class = sorted(feat_by_class)

            file = open(path_to_store_important_features_by_class_file, 'w')

            for f in her_and_ler:
                file.write(','.join(f) + '\n')
            file.close()


        else:
            print ("Length not equal, exiting...")
            sys.exit()


    def plot_feature_selection(self):


        transform = SelectPercentile(score_func=_score_func)

        clf = Pipeline([('anova', transform), ('clf', ExtraTreesClassifier())])

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

        plt.title(
            'Performance of the ExtraTree-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')

        plt.axis('tight')
        plt.show()


###############
# Set paths
###############

path_to_labelled_file = 'path_to_labelled_file'

path_to_store_vocabulary_file = 'path_to_store_vocabulary_file'
path_to_store_feature_selection_boolean_file = 'path_to_store_feature_selection_boolean_file'
path_to_store_complete_feature_importance_file = 'path_to_store_complete_feature_importance_file'
path_to_store_top_important_features_file = 'path_to_store_top_important_features_file'
path_to_store_feature_importance_file = 'path_to_store_feature_importance_file'
path_to_store_important_features_by_class_file = 'path_to_store_important_features_by_class_file'


# hyperparameters (for non-grid-search)

_n_estimators = 10
_criterion = "gini"
_max_depth = None
_min_samples_split = 2
_percentile = 85
_score_func = chi2
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

    et = ExtraTree()

    ###################
    # select (uncomment) one of the methods to split data using Cross Validation
    ###################

    x_train,x_test,y_train,y_test = et.train_test_split()
    #x_train,x_test,y_train,y_test = et.stratified_shufflesplit()
    #x_train,x_test,y_train,y_test = et.stratified_kfolds()


    ##################
    # run ExtraTree Classifier
    ##################

    clf = et.train_classifier()


    ###################
    # use pipeline
    ###################

    # clf = et.use_pipeline()

    ###################
    # use pipeline and use feature selection
    ###################

    # clf = et.use_pipeline_with_fs()


    ###################
    # Get feature importance
    ###################

    et.get_important_features(clf)


    ##################
    # Plot feature selection
    ##################

    # et.plot_feature_selection()
