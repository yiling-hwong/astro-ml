__author__ = 'yi-linghwong'

import sys
import os
import operator
import pandas as pd
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
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
import scipy as sp
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt


class NaiveBayes():


    def train_test_split(self):

        #################
        #Split the dataset in training and test set:
        #################

        docs_train, docs_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        print ("Number of data point is "+str(len(y)))

        return docs_train, docs_test, y_train, y_test


    def stratified_shufflesplit(self):

        ####################
        # Stratified ShuffleSplit cross validation iterator
        # Provides train/test indices to split data in train test sets.
        # This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds.
        # The folds are made by preserving the percentage of samples for each class.
        ####################


        sss = StratifiedShuffleSplit(y, 5, test_size=0.2, random_state=42)

        for train_index, test_index in sss:
            print("TRAIN:", train_index, "TEST:", test_index)
            docs_train, docs_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return docs_train, docs_test, y_train, y_test


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
            print("TRAIN:", train_index, "TEST:", test_index)
            docs_train, docs_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return docs_train, docs_test, y_train, y_test


    def train_classifier(self):

        # Get list of features
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=_ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation###

        tfidf_transformer = TfidfTransformer(use_idf = _use_idf)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        # train the classifier

        print ("Fitting data ...")
        clf = MultinomialNB(alpha=_alpha).fit(X_tfidf, y_train)


        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, X_tfidf, y_train, cv=10, scoring='f1_weighted')
        print ("Cross validation score: "+str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation
        # the factor two is to signify 2 sigma, which is 95% confidence level

        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


        ##################
        # run classifier on test data
        ##################

        X_test_CV = count_vect.transform(docs_test)

        print ("Shape of test data is "+str(X_test_CV.shape))

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        y_predicted = clf.predict(X_test_tfidf)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score on test data is: %0.2f " % clf.score(X_test_tfidf,y_test))

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf, count_vect


    def train_classifier_use_feature_selection(self):

        # Get list of features
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=_ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation###

        tfidf_transformer = TfidfTransformer(use_idf=_use_idf)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        #################
        # feature selection
        #################

        selector = SelectPercentile(score_func=_score_func, percentile=_percentile)

        print ("Fitting data with feature selection ...")
        selector.fit(X_tfidf, y_train)

        # get how many features are left after feature selection
        X_features = selector.transform(X_tfidf)

        print ("Shape of array after feature selection is "+str(X_features.shape))

        clf = MultinomialNB(alpha=_alpha).fit(X_features, y_train)

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


        ####################
        #test clf on test data
        ####################

        X_test_CV = count_vect.transform(docs_test)

        print ("Shape of test data is "+str(X_test_CV.shape))

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        # apply feature selection on test data too
        X_test_selector = selector.transform(X_test_tfidf)
        print ("Shape of array for test data after feature selection is "+str(X_test_selector.shape))

        y_predicted = clf.predict(X_test_selector)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score on test data is: %0.2f " % clf.score(X_test_selector,y_test))


        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        return clf, count_vect


    def use_pipeline(self):

        #####################
        #Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([
                ('vect', TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=0.90)),
                ('clf', MultinomialNB()),
        ])


        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'vect__ngram_range': [(1,1),(1,2), (1,3)],
            'vect__use_idf': (True, False),
            'clf__alpha': (0.4, 0.5)
        }

        #################
        # Exhaustive search over specified parameter values for an estimator, use cv to generate data to be used
        # implements the usual estimator API: when “fitting” it on a dataset all the possible combinations of parameter values are evaluated and the best combination is retained.
        #################

        cv = StratifiedShuffleSplit(y_train, n_iter=5, test_size=0.2, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=cv, n_jobs=-1)
        clf_gs = grid_search.fit(docs_train, y_train)

        ###############
        # print the cross-validated scores for the each parameters set explored by the grid search
        ###############

        best_parameters, score, _ = max(clf_gs.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        print("Score for gridsearch is %0.2f" % score)

        #y_predicted = clf_gs.predict(docs_test)


        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        ngram_range = best_parameters['vect__ngram_range']
        use_idf = best_parameters['vect__use_idf']
        alpha = best_parameters['clf__alpha']

        # vectorisation

        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation

        tfidf_transformer = TfidfTransformer(use_idf=use_idf)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        # train the classifier

        print ("Fitting data with best parameters ...")
        clf = MultinomialNB(alpha=alpha).fit(X_tfidf, y_train)

        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, X_tfidf, y_train, cv=10, scoring='f1_weighted')
        print ("Cross validation score: "+str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation

        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        ##################
        # run classifier on test data
        ##################

        X_test_CV = count_vect.transform(docs_test)

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        y_predicted = clf.predict(X_test_tfidf)

        # print the mean accuracy on the given test data and labels

        print ("Classifier score on test data is: %0.2f " % clf.score(X_test_tfidf,y_test))


        # Print and plot the confusion matrix

        print(metrics.classification_report(y_test, y_predicted))
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        # import matplotlib.pyplot as plt
        # plt.matshow(cm)
        # plt.show()

        return clf,count_vect


    def use_pipeline_with_fs(self):

        #####################
        #Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
        #####################

        pipeline = Pipeline([
                ('vect', TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=0.90)),
                ("selector", SelectPercentile()),
                ('clf', MultinomialNB()),
        ])


        # Build a grid search to find the best parameter
        # Fit the pipeline on the training set using grid search for the parameters
        parameters = {
            'vect__ngram_range': [(1,1), (1,2), (1,3)],
            'vect__use_idf': (True, False),
            'selector__score_func': (chi2, f_classif),
            'selector__percentile': (85, 95, 100),
            'clf__alpha': (0.4, 0.5)
        }

        #################
        # Exhaustive search over specified parameter values for an estimator, use cv to generate data to be used
        # implements the usual estimator API: when “fitting” it on a dataset all the possible combinations of parameter values are evaluated and the best combination is retained.
        #################

        cv = StratifiedShuffleSplit(y_train, n_iter=5, test_size=0.2, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=cv, n_jobs=-1)
        clf_gs = grid_search.fit(docs_train, y_train)

        ###############
        # print the cross-validated scores for the each parameters set explored by the grid search
        ###############

        best_parameters, score, _ = max(clf_gs.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        print("Score for gridsearch is %0.2f" % score)

        #y_predicted = clf_gs.predict(docs_test)

        ###############
        # run the classifier again with the best parameters
        # in order to get 'clf' for get_important_feature function!
        ###############

        ngram_range = best_parameters['vect__ngram_range']
        use_idf = best_parameters['vect__use_idf']
        score_func = best_parameters['selector__score_func']
        percentile = best_parameters['selector__percentile']
        alpha = best_parameters['clf__alpha']

        # vectorisation

        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation

        tfidf_transformer = TfidfTransformer(use_idf=use_idf)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        #################
        # feature selection
        #################

        selector = SelectPercentile(score_func=score_func, percentile=percentile)

        combined_features = Pipeline([
                                        ("vect", count_vect),
                                        ("tfidf", tfidf_transformer),
                                        ("feat_select", selector)
        ])

        X_features = combined_features.fit_transform(docs_train,y_train)
        X_test_features = combined_features.transform(docs_test)

        print ("Shape of train data after feature selection is "+str(X_features.shape))
        print ("Shape of test data after feature selection is "+str(X_test_features.shape))


        # run classifier on selected features

        clf = MultinomialNB(alpha=alpha).fit(X_features, y_train)

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

        # import matplotlib.pyplot as plt
        # plt.matshow(cm)
        # plt.show()

        return clf, count_vect


    def predict_posts(self):

        docs_train, docs_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

        print("Number of data point is " + str(len(y)))

        ###############
        # uncomment either one of the below
        # predict unlabelled tweet OR test classifier on gold standard
        ###############

        # dataset_topredict = pd.read_csv(path_to_file_to_be_predicted, header=0, names=['tweets'])
        dataset_topredict = pd.read_csv(path_to_gold_standard_file, header=0, names=['tweets', 'class'])

        X_topredict = dataset_topredict['tweets']
        y_goldstandard = dataset_topredict['class']

        ###############
        # train classifier
        ###############

        # Get list of features
        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=_ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print("Shape of train data is " + str(X_CV.shape))

        # tfidf transformation###

        tfidf_transformer = TfidfTransformer(use_idf=_use_idf)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)

        # train the classifier

        print("Fitting data ...")
        clf = MultinomialNB(alpha=_alpha).fit(X_tfidf, y_train)

        ##################
        # get cross validation score
        ##################

        scores = cross_val_score(clf, X_tfidf, y_train, cv=10, scoring='f1_weighted')
        print("Cross validation score: " + str(scores))

        # Get average performance of classifier on training data using 10-fold CV, along with standard deviation
        # the factor two is to signify 2 sigma, which is 95% confidence level

        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        ##################
        # run classifier to predict tweets
        ##################

        X_test_CV = count_vect.transform(X_topredict)

        print("Shape of test data is " + str(X_test_CV.shape))

        X_test_tfidf = tfidf_transformer.transform(X_test_CV)

        y_predicted = clf.predict(X_test_tfidf)

        ##################
        # run classifier on gold standard (tweets that were labelled by twitter insight)
        ##################

        # print the mean accuracy on the given test data and labels

        print("Classifier score on test data is: %0.2f " % clf.score(X_test_tfidf, y_goldstandard))

        print(metrics.classification_report(y_goldstandard, y_predicted))
        cm = metrics.confusion_matrix(y_goldstandard, y_predicted)
        print(cm)

        ##################
        # write prediction results to file
        ##################

        f = open(path_to_store_predicted_results, 'w')

        for yp in y_predicted:
            f.write(yp + '\n')

        f.close()


    def cv_and_train(self):

        ################
        #Stratified ShuffleSplit cross validation iterator
        #Provides train/test indices to split data in train test sets.
        #This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds.
        #The folds are made by preserving the percentage of samples for each class.
        ################


        sss = StratifiedShuffleSplit(y, 5, test_size=0.2, random_state=42)

        print (len(sss))
        print (sss)

        for train_index, test_index in sss:
            #print("TRAIN:", train_index, "TEST:", test_index)
            docs_train, docs_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=(1,3))
            X_CV = count_vect.fit_transform(docs_train)
            ### print number of unique words (n_features)
            print (X_CV.shape)
            tfidf_transformer = TfidfTransformer()
            X_tfidf = tfidf_transformer.fit_transform(X_CV)

            clf = MultinomialNB(alpha=0.5).fit(X_tfidf, y_train)

            print ("Fitting data ...")
            clf.fit(X_tfidf, y_train)

            X_test_CV = count_vect.transform(docs_test)
            X_test_tfidf = tfidf_transformer.transform(X_test_CV)

            y_predicted = clf.predict(X_test_tfidf)

            scores = cross_val_score(clf, X_test_tfidf, y_test, cv=3, scoring='f1_weighted')
            print ("Cross validation score:%s " % scores)

            print ("Classifier score is: %s " % clf.score(X_test_tfidf,y_test))
            print(metrics.classification_report(y_test, y_predicted))
            cm = metrics.confusion_matrix(y_test, y_predicted)
            print(cm)


    def get_important_features(self,clf,count_vect):

        ################
        #Prints features with the highest coefficient values, per class
        ################

        n=10

        class_labels = clf.classes_
        fb_her=clf.feature_log_prob_[0] ##feature probability for HER
        fb_ler=clf.feature_log_prob_[1] ##feature probability for LER
        feature_names = count_vect.get_feature_names()

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

        # write to file for normalisation

        f = open(path_to_store_feat_imp_for_normalisation,'a')

        f.write('\n')

        for imf in imp_feat:
            f.write(imf+'\n')

        f.close()

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

        # vectorisation

        count_vect = CountVectorizer(stop_words=stopwords, min_df=3, max_df=0.90, ngram_range=_ngram_range)
        X_CV = count_vect.fit_transform(docs_train)

        # print number of unique words (n_features)
        print ("Shape of train data is "+str(X_CV.shape))

        # tfidf transformation

        tfidf_transformer = TfidfTransformer(use_idf=_use_idf)
        X_tfidf = tfidf_transformer.fit_transform(X_CV)


        transform = SelectPercentile(score_func=_score_func)

        clf = Pipeline([('anova', transform), ('clf', MultinomialNB(alpha=_alpha))])

        ###############################################################################
        # Plot the cross-validation score as a function of percentile of features
        score_means = list()
        score_stds = list()
        percentiles = (10, 20, 30, 40, 60, 80, 85, 95, 100)

        for percentile in percentiles:
            clf.set_params(anova__percentile=percentile)
            # Compute cross-validation score using all CPUs
            this_scores = cross_validation.cross_val_score(clf, X_tfidf, y_train, n_jobs=-1)
            score_means.append(this_scores.mean())
            score_stds.append(this_scores.std())


        plt.errorbar(percentiles, score_means, np.array(score_stds))

        plt.title(
            'Performance of the NB-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')

        plt.axis('tight')
        plt.show()


###############
# variables
###############

path_to_labelled_file = 'path_to_labelled_file'
path_to_stopword_file = 'path_to_stopword_file'

path_to_file_to_be_predicted = 'path_to_file_to_be_predicted'
path_to_gold_standard_file = 'path_to_gold_standard_file'
path_to_store_predicted_results = 'path_to_store_predicted_results '

path_to_store_features_by_probability_file = 'path_to_store_features_by_probability_file'
path_to_store_feature_selection_boolean_file = 'path_to_store_feature_selection_boolean_file'
path_to_store_list_of_feature_file = 'path_to_store_list_of_feature_file'
path_to_store_coefficient_file = 'path_to_store_coefficient_file'
path_to_store_feature_log_prob_for_class_0 = 'path_to_store_feature_log_prob_for_class_0' #Empirical log probability of features given a class
path_to_store_feature_log_prob_for_class_1 = 'path_to_store_feature_log_prob_for_class_1'
path_to_store_important_features_by_class_file = 'path_to_store_important_features_by_class_file'

path_to_store_feat_imp_for_normalisation = 'path_to_store_feat_imp_for_normalisation'


# for classifier without pipeline
_ngram_range = (1,1)
_alpha = 0.6
_use_idf = False
_percentile = 85
_score_func = f_classif



def get_data_set():

    #############
    # Get dataset
    #############

    dataset = pd.read_csv(path_to_labelled_file, header=0, names=['posts', 'class'])

    X = dataset['posts']
    y = dataset['class']

    return X,y

def get_stop_words():

    ###########
    # get stopwords
    ###########

    lines = open(path_to_stopword_file, 'r').readlines()

    my_stopwords=[]
    for line in lines:
        my_stopwords.append(line.replace("\n", ""))

    stopwords = text.ENGLISH_STOP_WORDS.union(my_stopwords)

    return stopwords


if __name__ == '__main__':

    X = get_data_set()[0]
    y = get_data_set()[1]
    stopwords = get_stop_words()

    nb = NaiveBayes()

    ###################
    # select one of the method to split data using Cross Validation
    ###################

    docs_train,docs_test,y_train,y_test = nb.train_test_split()
    #docs_train,docs_test,y_train,y_test = nb.stratified_shufflesplit()
    #docs_train,docs_test,y_train,y_test = nb.stratified_kfolds()


    ##################
    # run NB Classifier
    ##################

    #clf, count_vect = nb.train_classifier()


    ###################
    # run NB Classifier and use feature selection
    ###################

    #clf, count_vect = nb.train_classifier_use_feature_selection()


    ###################
    # use pipeline
    ###################

    clf, count_vect = nb.use_pipeline()

    ###################
    # use pipeline and use feature selection
    ###################

    #clf, count_vect = nb.use_pipeline_with_fs()

    ###################
    # Get feature importance
    ###################

    nb.get_important_features(clf,count_vect)


    ###################
    # Run classifier and then predict tweets
    ###################

    #nb.predict_posts()


    ##################
    # CV and train: run a for loop through CV data train for each loop
    # useful helper function to compare results of classifier at each loop
    ##################

    #nb.cv_and_train()


    ##################
    # Plot feature selection
    ##################

    #nb.plot_feature_selection()


