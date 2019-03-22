import re
import csv
import numpy as np

"""
Set of functions to clean and preprocess
tweet files.
"""

def sentimentLabeler(sentiment):
    """
    Return sentiments as int labels.
    @sentiment: String repr. tweet sentiment.
    """
    # don't use StringIndexer bcos lbls will depend on freq
    if sentiment == "negative":
        return 0
    elif sentiment == "positive":
        return 1
    else:
        return 2


def tweetCleaner(tweet):
    """
    Return a cleaned tweet.
    @tweet: String repr. a tweet.
    """
    cleaned_tweet = tweet.encode('unicode_escape').decode('utf-8')  # expose unicode characters
    cleaned_tweet = re.sub(r'\\n+', ' ', cleaned_tweet)  # rmv new lines
    cleaned_tweet = re.sub('^RT\s@\w+:\s', '', cleaned_tweet)  # rmv retweets
    cleaned_tweet = re.sub('https?://\w+\.\w+\/[\w|\.]*\s?|https?', '', cleaned_tweet)  # rmv weblinks
    cleaned_tweet = re.sub(r'\\[u|U|x]\w+', '',cleaned_tweet)  # rmv unicode chars
    # rmv html codes e.g &amp, &gt
    cleaned_tweet = re.sub(r'\&[\w]+', '', cleaned_tweet)
    cleaned_tweet = re.sub(r'\#[\w]+', '', cleaned_tweet)  # rmv hashtags
    # rmv foreign symbols e.g brackets etc
    cleaned_tweet = re.sub(r'[\[|\]|\(|\)|\%|\$|\*|\"]*', '', cleaned_tweet)

    # convert usernames to keyword
    cleaned_tweet = re.sub(r'\@[\w]+', '', cleaned_tweet)  #TODO: Remove as it affects gensim

    # replace stop words
    cleaned_tweet = re.sub(r'\@', 'at', cleaned_tweet)
    cleaned_tweet = re.sub(r'\&', 'and', cleaned_tweet)

    # break abbrevations & remove apostrophes
    cleaned_tweet = re.sub('\'d', ' would', cleaned_tweet)
    cleaned_tweet = re.sub('\'m', ' am', cleaned_tweet)
    cleaned_tweet = re.sub('n\'t', ' not', cleaned_tweet)
    cleaned_tweet = re.sub('\'re', ' are', cleaned_tweet)
    cleaned_tweet = re.sub('\'ll', ' will', cleaned_tweet)
    cleaned_tweet = re.sub(r'\'', '', cleaned_tweet)

    cleaned_tweet = ' '.join(cleaned_tweet.split())  # rmv trailing spaces
    return cleaned_tweet


def importTweets(filename):
    """
    Return an array of cleaned tweets and a numpy vector of labels.
    """
  #  with open(filename, 'r') as tfile:
    with open(filename, encoding="utf-8") as tfile:
        data, lbls = [], []
        input_reader = csv.reader(tfile)
        next(input_reader, None)  # skip header

        for row in input_reader:
            txt, lbl = tweetCleaner(row[10]), sentimentLabeler(row[1])
            data.append(txt)
            lbls.append(lbl)

    tfile.close()

    # convert to matrix
    data = np.asarray(data)
    lbls = np.asarray(lbls)

    # shuffle data
    data_len = lbls.shape[0]
    rand_idxs = np.random.permutation(data_len)
    data, lbls = data[rand_idxs], lbls[rand_idxs]

    data = data.tolist() #iterable. Needed for vectorizer

    # split data into train:test::80:20
    split_idx = round(0.80 * data_len)
    data_train, y_train, data_test, y_test = data[0:split_idx], lbls[0:split_idx], data[split_idx:], lbls[split_idx:]
    return data_train, y_train, data_test, y_test
