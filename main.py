"""
Test doc2vec against tf-idf.
"""
from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import gensim as gn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from preprocessor import importTweets

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

############################## helper functions here ###############################################

def trim(s):
    """
    Trim string to fit on terminal (assuming 80-column display)
    """
    return s if len(s) <= 80 else s[:77] + "..."

def benchmarkTfIdf(clf):
    """
    Benchmark classifiers using TF-IDF
    """
    print('_' * 80)
    print("Training Tf-Idf: ")
    print(clf)
    t0 = time()
    clf.fit(x_train0, y_train) #! train data using x(features) & y(targets)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred0 = clf.predict(x_test0)   #!predict data
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred0) #! check the accuracy of preds using the test targets
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report:")
    print(metrics.classification_report(y_test, pred0, target_names=class_names))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred0))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

def benchmarkDoc2Vec(clf):
    """
    Benchmark classifiers
    """
    print('_' * 80)
    print("Training Doc2Vec: ")
    print(clf)
    t0 = time()
    clf.fit(x_train1, y_train) #! train data using x(features) & y(targets)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred1 = clf.predict(x_test1)   #!predict data
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred1) #! check the accuracy of preds using the test targets
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report:")
    print(metrics.classification_report(y_test, pred1, target_names=class_names))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred1))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

############################## run experiments here ###############################################

# import and clean tweets
print("Importing training data from the file")
class_names = ['Negative sentiment', 'Positive sentiment', 'Neutral sentiment'] # see sentimentLabeler() in preprocessor
#TODO: Remove %USERNAME% as it affects gensim
data_train, y_train, data_test, y_test = importTweets('ProductReview.csv')

# initialize tf-idf vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

# extract tf-idf features and print time taken & no. of samples
print("Extracting Tf-Idf features from the training data")
t0 = time()
x_train0 = vectorizer.fit_transform(data_train)
duration = time() - t0
print("done in %fs" % (duration))
print("n_samples: %d, n_features: %d" % x_train0.shape)
print()

print("Extracting Tf-Idf features from the test data")
t0 = time()
x_test0 = vectorizer.transform(data_test)
duration = time() - t0
print("done in %fs" % (duration))
print("n_samples: %d, n_features: %d" % x_test0.shape)
print()

print(x_train0.shape)

# get the names of the tf-idf features
feature_names = vectorizer.get_feature_names()
if feature_names:
    feature_names = np.asarray(feature_names)

doc_train = []
for i in range(len(data_train)):
    token = gn.utils.simple_preprocess(data_train[i])
    doc_train.append(gn.models.doc2vec.TaggedDocument(token, [i]))

doc_test = []
for i in range(len(data_test)):
    token = gn.utils.simple_preprocess(data_test[i])
    doc_test.append(token)

print("Extracting Doc2Vec features from the training data")
t0 = time()
model = gn.models.doc2vec.Doc2Vec(vector_size=100, min_count=1, epochs=1000)
model.build_vocab(doc_train)
model.train(doc_train, total_examples=model.corpus_count, epochs=model.epochs)
x_train1 = np.asarray([model.docvecs[i].tolist() for i in range(len(model.docvecs))])
duration = time() - t0
print("done in %fs" % (duration))
print("n_samples: %d, n_features: %d" % x_train1.shape)
print()

print("Extracting Doc2Vec features from the test data")
t0 = time()
x_test1 = np.asarray([model.infer_vector(doc_test[i]).tolist() for i in range(len(doc_test))])
duration = time() - t0
print("done in %fs" % (duration))
print("n_samples: %d, n_features: %d" % x_test1.shape)
print()

# run classifiers
results = []
for penalty in ["l2", "l1"]:
     print('=' * 80)
     print("%s penalty" % penalty.upper())
     # Train Logistic reg model
     results.append(benchmarkTfIdf(LogisticRegression(multi_class='multinomial',penalty=penalty, solver='saga', tol=0.1)))
     results.append(benchmarkDoc2Vec(LogisticRegression(multi_class='multinomial',penalty=penalty, solver='saga', tol=0.1)))



# Train sparse K nearest neighbour classifiers
print('=' * 80)
print("K-nearest neighbor")
results.append(benchmarkTfIdf(KNeighborsClassifier()))
results.append(benchmarkDoc2Vec(KNeighborsClassifier()))



# # #make some plots
# indices = np.arange(len(results))
#
# results = [[x[i] for x in results] for i in range(4)]
#
# clf_names, score, training_time, test_time = results
# training_time = np.array(training_time) / np.max(training_time)
# test_time = np.array(test_time) / np.max(test_time)
#
# plt.figure(figsize=(12, 8))
# plt.title("Score")
# plt.barh(indices, score, .2, label="score", color='navy')
# plt.barh(indices + .3, training_time, .2, label="training time",
#           color='c')
# plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
# plt.yticks(())
# plt.legend(loc='best')
# plt.subplots_adjust(left=.25)
# plt.subplots_adjust(top=.95)
# plt.subplots_adjust(bottom=.05)
#
# for i, c in zip(indices, clf_names):
#      plt.text(-.3, i, c)
#
# plt.show()