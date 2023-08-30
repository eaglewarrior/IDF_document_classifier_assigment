from nltk.corpus import stopwords
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

def clean_text(text_data_df, col):
  for i in range(0,len(text_data_df[col])):
    if type(text_data_df.iloc[i][col]) != float:
      text_data_df.iloc[i][col] = text_data_df.iloc[i][col].lower().replace("\n"," ").replace("\t"," ").strip(" ")
      text_data_df.iloc[i][col] = "".join(c for c in text_data_df.iloc[i][col] if c not in punct)
      filtered_words = [w for w in text_data_df[col].iloc[i].split() if w not in stopwords.words('english')]
      # individual text database-oriented keywords list is best but due to time constraints could not build it 
      text_data_df.iloc[i][col] = " ".join([c for c in text_data_df[col].iloc[i].split(" ") if not(c[:1].isdigit() and c[1:2] in (p for p in punct))])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split() if w not in stop_words])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split() if w[:-1] not in stop_words])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split() if w not in geo_words])  
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split() if w[:1] not in list(map(lambda x: str(x),range(3))) and w[:1] not in list(map(lambda x: str(x),range(4,10)))])  
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split(" ") if not(w[:1].isdigit() and w[1:].isalpha())])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split(" ") if not(w[:3].isdigit() and w[3:].isalpha())])
      text_data_df.iloc[i][col] = " ".join([w[:-1] if not(w[:1].isdigit()) and w.endswith(".") else w for w in text_data_df.iloc[i][col].split(" ")])
      text_data_df.iloc[i][col] = " ".join([w for w in text_data_df[col].iloc[i].split(" ") if len(w) > 2 and len(w) < 15])
      text_data_df.iloc[i][col] = " ".join([w.replace("."," ") if len(w) > 9 or len(w) < 7 else w for w in text_data_df[col].iloc[i].split(" ") ])
      tag_map = defaultdict(lambda : wn.NOUN)
      tag_map['J'] = wn.ADJ
      tag_map['V'] = wn.VERB
      tag_map['R'] = wn.ADV
      for i in range(len(text_data_df[col])):
          lemma = []
      #     text_lemma = ""
          text_tokens = word_tokenize(text_data_df.iloc[i][col])
          lemma_function = WordNetLemmatizer()
          for token, tag in pos_tag(text_tokens):
              lemma.append(lemma_function.lemmatize(token, tag_map[tag[0]]))
          text_data_df.iloc[i][col] = " ".join(l for l in lemma )
  return text_data_df
