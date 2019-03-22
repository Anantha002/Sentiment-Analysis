About Project

This project evaluates the performance of Paragraph vectors (Doc2Vec) against feature vectors that are derived from traditional NLP feature extraction algorithms like Term Frequency-Inverse Document Frequency (TF-IDF). The comparison is based on a 3-class (positive, negative, & neutral) sentiment analysis of tweets and product reviews.

TF-IDF generates large sparse vectors that do not take into account the ordering of words within a text whereas Doc2Vec generates dense smaller vectors that take such ordering into account. The initial hypothesis is that Doc2Vec will outperform TF-IDF on a multiclass sentiment task because Doc2Vec mitigates the  short comings of TF-IDF. This experiment evaluates the performance of Doc2Vec and TF-IDF for multiclass sentiment analysis of tweets and product reviews datasets.

To compare both feature extractors, two different classifiers are used: logistic regression and k-nearest neighbor. This is implemented in order to eliminate any potential bias on how the classifiers work since one classifier is a parametric classifier while the other is a non-parametric classifier. By comparing the two feature extractors for multiclass sentiment analysis, it is possible to determine if the benefits brought by Doc2Vec translate to an improved performance over the more traditional NLP approaches to feature extraction.

Datasets:
The datasets for this project are
A)Twitter Data (Tweets)
B)Product Reviews


Procedure:

Both datasets are correctly labeled with the correct sentiments. Unwanted attributes and characters will reduce the performance of algorithm on sentiment analysis. So this will be removed by using Python regular expression library. Once the tweets were cleaned, we permutated their order in order to randomize the data and then we split the entire dataset into training and test datasets, with a ratio of 4:1 respectively. After splitting the dataset, we completed the preprocessing step by converting the sentiment labels into numeric values. Specifically, we assigned negative sentiments a value of 0, positive sentiments a value of 1, and neutral sentiments a value of 2. Two python libraries are used for each of our feature extractors: sklearn and gensim. Then for classification, sklearn machine learning library is implemented. Finally, classification report showing the classifier’s recall, precision, and f1-score is generated along with a confusion matrix of its predictions.


Instruction to run the code:
1)In main.py, line 115 has the input parameter for dataset. In this folder there are 2 datasets;Tweets and Product Reviews. Choose any one of your choice and update in line no 115.
2)Make sure all the required libaries are imported.
3)Run the program, for each dataset it take around 15 mins to compute the classification report for TF-IDF and Doc2Vec.
4)See the console for output to compare the performance.

