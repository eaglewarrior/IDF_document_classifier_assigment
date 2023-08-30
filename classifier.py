import sklearn
import string, re
import pandas as pd, numpy as np
import PyPDF2
import os, pickle
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics     import accuracy_score,confusion_matrix,f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from collections import Counter


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

count_vectorizer = CountVectorizer(strip_accents='ascii',lowercase=True, analyzer='word', max_df=0.25, min_df=0.05, ngram_range=(2, 2))

def get_form_or_not(df, col):
  X_train_cv = count_vectorizer.fit_transform(df['col'])
  X_train_cv = X_train_cv.toarray()
  X_train_cv['filename']=df['filename']
  ref = X_train_cv[X_train_cv.filename=='
  
  
