'''

Class for emotional classification of a given query text
@author : debarghya nandi

'''

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, log_loss
import os
import string
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.classifiers import NaiveBayesClassifier
import pandas as pd
import numpy as np

DATA_FOLDER= 'data/parsed2_imsdb'




'''
A class that performs emotional classification
The class uses the ISEAR dataset as the training model for predicting the emotional tag for a given text
It also returns the probabilistic value for any given text
'''


class emotion_classify:
    def __init__(self):
        self.df = pd.read_csv('isear.csv', delimiter = '|')
        self.a = pd.Series(self.df['joy'])
        self.b = pd.Series(self.df['During the period of falling in love each time that we met and especially when we had not met for a long time.'])
        self.new_df = pd.DataFrame({'Text': self.b, 'Emotion': self.a})

        self.stop = set(stopwords.words('english'))  ## stores all the stopwords in the lexicon
        self.exclude = set(string.punctuation)  ## stores all the punctuations
        self.lemma = WordNetLemmatizer()

        ## lets create a list of all negative-words
        self.negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                         'even though', 'yet']

        ## create a separate list to store texts and emotion
        self.em_list = []
        self.text_list = []

        ## create the training set
        self.train = []

         # stores the summarized text in a list
        self.sum_text_list = []

        # the e-score list stores the e-score for each document
        self.e_score_dict = {}

        # call the driver function
        self.main()

    '''
    A function for cleaning up all the documents
    # removes stop words
    # removes punctuations
    # uses lemmatizer
    '''

    def clean(self, doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in self.stop if i not in self.negative])
        punc_free = "".join([ch for ch in stop_free if ch not in self.exclude])
        normalized = " ".join([self.lemma.lemmatize(word) for word in punc_free.split()])
        return normalized

    '''
    Function to iterate and clean up all texts
    '''

    def iterate_clean(self):
        for i in range(self.df.shape[0]):
            self.new_df.loc[i]['Text'] = self.clean(self.new_df.loc[i]['Text'])

    '''
    Function to iterate and populate text list
    '''

    def iterate_pop_text(self):
        for i in range(self.new_df.shape[0]):
            self.text_list.append(self.new_df.loc[i]['Text'])

    '''
    Function to iterate and populate emotion list
    '''

    def iterate_pop_emotion(self):
        for i in range(self.new_df.shape[0]):
            self.em_list.append(self.new_df.loc[i]['Emotion'])

    '''
    Function to create training set
    '''

    def create_train(self):
        for i in range(self.new_df.shape[0]):
            self.train.append([self.text_list[i], self.em_list[i]])

    '''
    Function to create model
    classify the query text
    and then summarize other texts
    classify them and return a dictionary containing the e-score for all documents
    '''


    def classify_text(self):
        X_train, X_test, y_train, y_test = train_test_split(self.new_df['Text'], self.new_df['Emotion'],test_size=.01,  random_state = 0)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        clf = SGDClassifier(loss = 'log', alpha=0.0001, verbose = 1, early_stopping=True).fit(X_train_tfidf, y_train)
        print(type(clf.predict_proba(count_vect.transform(X_test))))
        for genre in list(os.listdir(DATA_FOLDER)):
            if not genre.endswith(".DS_Store"):
                genre_folder = DATA_FOLDER + '/' + genre
                for i, film in enumerate(list(os.listdir(genre_folder))):
                    if film.endswith('.txt'):
                        sentences = []
                        with open(genre_folder + '/' + film, 'r') as file:
                            sentences = file.read().splitlines()
                        sentences = sentences[15:]
                        preds = clf.predict_proba(count_vect.transform(sentences))
                        print(preds.shape)
                        np.savetxt('data/emotion_vectors3/' + genre + '/' + film,preds)




        # predictions = clf.predict(count_vect.transform(X_test))
        # print(clf.predict_proba(count_vect.transform(X_test))[:-10])
        # print(classification_report(y_test,predictions))

    '''
    A function which is the driver for the entire class

    '''

    def main(self):
        self.iterate_clean()
        self.iterate_pop_emotion()
        self.iterate_pop_text()
        self.create_train()
        self.classify_text()
emotion_classify()
