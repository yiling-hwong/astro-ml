__author__ = 'yi-linghwong'

import sys
import os
import operator
import pandas as pd
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


class NaiveBayes():


    def train_test_split(self):

        #################
        #Split the dataset in training and test set:
        #################

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        print ("Number of data point is "+str(len(y)))

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

        print (len(skf))
        print(skf)

        for train_index, test_index in skf:
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return x_train, x_test, y_train, y_test


    def train_classifier(self):


        print ("Fitting data ...")
        clf = BernoulliNB().fit(x_train, y_train)


        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='f1_weighted')
        print ("Cross validation score: "+str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation
        # the factor two is to signify 2 sigma, which is 95% confidence level

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
        #Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([
                ('clf', BernoulliNB()),
        ])


        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'clf__alpha': (0.3, 0.4, 0.5, 0.6)
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

        #y_predicted = clf_gs.predict(x_test)


        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        alpha = best_parameters['clf__alpha']


        # train the classifier

        print ("Fitting data with best parameters ...")
        clf = BernoulliNB(alpha=alpha).fit(x_train, y_train)

        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='f1_weighted')
        print ("Cross validation score: "+str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation

        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        ##################
        # run classifier on test data
        ##################

        y_predicted = clf.predict(x_test)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score on test data is: %0.2f " % clf.score(x_test,y_test))


        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        # import matplotlib.pyplot as plt
        # plt.matshow(cm)
        # plt.show()

        return clf


    def use_pipeline_with_fs(self):

        #####################
        #Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([
                ("selector", SelectPercentile()),
                ('clf', BernoulliNB()),
        ])


        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'selector__score_func': (chi2, f_classif),
            'selector__percentile': (85, 95, 100),
            'clf__alpha': (0.3, 0.4, 0.5, 0.6)
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

        #y_predicted = clf_gs.predict(x_test)

        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        score_func = best_parameters['selector__score_func']
        percentile = best_parameters['selector__percentile']
        alpha = best_parameters['clf__alpha']


        #################
        # feature selection
        #################

        selector = SelectPercentile(score_func=score_func, percentile=percentile)

        combined_features = Pipeline([
                                        ("feat_select", selector)
        ])

        X_features = combined_features.fit_transform(x_train,y_train)
        X_test_features = combined_features.transform(x_test)

        print ("Shape of train data after feature selection is "+str(X_features.shape))
        print ("Shape of test data after feature selection is "+str(X_test_features.shape))


        # run classifier on selected features

        clf = BernoulliNB(alpha=alpha).fit(X_features, y_train)

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


        #################
        # run classifier on test data
        #################


        y_predicted = clf.predict(X_test_features)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score on test data is: %0.2f " % clf.score(X_test_features,y_test))

        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf


    def get_important_features(self,clf):

        ################
        #Prints features with the highest coefficient values, per class
        ################

        n=40

        class_labels = clf.classes_
        fb_her=clf.feature_log_prob_[0] ##feature probability for HER
        fb_ler=clf.feature_log_prob_[1] ##feature probability for LER
        feature_names = get_data_set()[2][:-1]

        f=open(path_to_store_list_of_feature_file, 'w')

        for fn in feature_names:
            f.write(str(fn)+'\n')
        f.close()

        print (len(fb_her))
        print (len(fb_ler))
        print (len(feature_names))

        #################
        # if feature selection was used, need to find out which are the features that are retained
        #################

        if len(fb_her) != len(feature_names):

            print ()
            print ("###### feature selection was used, getting retained features ######")

            lines = open(path_to_store_feature_selection_boolean_file).readlines()

            feature_boolean = []

            for line in lines:
                spline = line.replace('\n','')
                feature_boolean.append(spline)

            if len(feature_boolean) == len(feature_names):

                selected_features = zip(feature_names,feature_boolean)

                feature_names = []

                for sf in selected_features:
                    if sf[1] == 'True':
                        feature_names.append(sf[0])

                print ("Length of retained features is "+str(len(feature_names)))

            else:
                print ("length not equal, exiting...")
                sys.exit()


        ################
        #the next two lines are for printing the highest feat_probability for each class
        ################

        topn_class1 = sorted(zip(fb_her, feature_names))[-n:]
        topn_class2 = sorted(zip(fb_ler, feature_names))[-n:]

        #################
        # Most important features are the ones where the difference between feat_prob are the biggest
        #################

        diff = [abs(a-b) for a,b in zip(fb_her,fb_ler)]


        # sort the list by the value of the difference, and return index of that element###
        sortli = sorted(range(len(diff)), key=lambda i:diff[i], reverse=True)[:200]


        # print out the feature names and their corresponding classes

        imp_feat=[]
        for i in sortli:

            if fb_her[i]>fb_ler[i]:
                imp_feat.append('HER,'+str(feature_names[i])+','+str(diff[i]))
            else:
                imp_feat.append('LER,'+str(feature_names[i])+','+str(diff[i]))


        #imp_feat=sorted(imp_feat)

        f4=open(path_to_store_important_features_by_class_file, 'w')

        for imf in imp_feat:
            f4.write(imf+'\n')

        f4.close()


        #################
        # Get features with highest probability
        #################


        coef=clf.coef_[0]

        coef_list=[]
        for c in coef:
            coef_list.append(c)

        if len(coef_list) == len(feature_names):

            feat_list=list(zip(coef_list,feature_names))

            feat_list.sort(reverse=True)

        else:
            print ("Length of coef and feature list not equal, exiting...")
            sys.exit()

        f=open(path_to_store_features_by_probability_file,'w')

        for fl in feat_list:
            f.write(str(fl)+'\n')

        f.close()



        ###############
        # HELPER FILES
        ###############

        f1=open(path_to_store_coefficient_file, 'w')

        for c in clf.coef_[0]:
            f1.write(str(c)+'\n')

        f1.close()

        f2=open(path_to_store_feature_log_prob_for_class_0, 'w')
        for fb in clf.feature_log_prob_[0]:
            f2.write(str(fb)+'\n')
        f2.close()

        f3=open(path_to_store_feature_log_prob_for_class_1, 'w')
        for fb in clf.feature_log_prob_[1]:
            f3.write(str(fb)+'\n')
        f3.close()


    def plot_feature_selection(self):


        transform = SelectPercentile(score_func=_score_func)

        clf = Pipeline([('anova', transform), ('clf', BernoulliNB(alpha=_alpha))])

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
            'Performance of the NB-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')

        plt.axis('tight')
        plt.show()


################
# Set paths
################

path_to_labelled_file = 'path_to_labelled_file'

path_to_store_features_by_probability_file = 'path_to_store_features_by_probability_file'
path_to_store_feature_selection_boolean_file = 'path_to_store_feature_selection_boolean_file'
path_to_store_list_of_feature_file = 'path_to_store_list_of_feature_file'
path_to_store_coefficient_file = 'path_to_store_coefficient_file'
path_to_store_feature_log_prob_for_class_0 = 'path_to_store_feature_log_prob_for_class_0' #Empirical log probability of features given a class
path_to_store_feature_log_prob_for_class_1 = 'path_to_store_feature_log_prob_for_class_1'
path_to_store_important_features_by_class_file = 'path_to_store_important_features_by_class_file'



# hyperparameters (for non-grid-search)

_alpha = 0.6
_percentile = 85
_score_func = f_classif



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

    nb = NaiveBayes()


    ###################
    # select one of the method to split data using Cross Validation
    ###################

    x_train,x_test,y_train,y_test = nb.train_test_split()
    #x_train,x_test,y_train,y_test = nb.stratified_shufflesplit()
    #x_train,x_test,y_train,y_test = nb.stratified_kfolds()


    ##################
    # run NB Classifier
    ##################

    clf = nb.train_classifier()


    ###################
    # use pipeline
    ###################

    #clf = nb.use_pipeline()

    ###################
    # use pipeline and use feature selection
    ###################

    #clf = nb.use_pipeline_with_fs()


    ###################
    # Get feature importance
    ###################

    nb.get_important_features(clf)


    ##################
    # Plot feature selection
    ##################

    #nb.plot_feature_selection()


