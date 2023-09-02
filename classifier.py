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
    ref_formd = X_train_cv[X_train_cv.filename=='00049397 (1).pdf'].iloc[:,:-1]
    ref_13F = X_train_cv[X_train_cv.filename=='01071683.pdf'].iloc[:,:-1]
    ref_82sfc = X_train_cv[X_train_cv.filename=='01074959.pdf'].iloc[:,:-1]
    ref_11k = X_train_cv[X_train_cv.filename=='01066872.pdf'].iloc[:,:-1]
    ref_X17as = X_train_cv[X_train_cv.filename=='01031026.pdf'].iloc[:,:-1]
    ref_TA2 = X_train_cv[X_train_cv.filename=='01029504.pdf'].iloc[:,:-1]
    
    result_13f =[]
    result_82sfc =[]
    result_11k =[]
    result_formd =[]
    result_X17as =[]
    result_TA2 =[]
    for f in  X_train_cv['filename']:
        result_13f.append(get_cosine(ref_13F, X_train_cv[X_train_cv.filename==f].iloc[:,:-1]))
        result_82sfc.append(get_cosine(ref_82sfc, X_train_cv[X_train_cv.filename==f].iloc[:,:-1]))
        result_11k.append(get_cosine(ref_11k, X_train_cv[X_train_cv.filename==f].iloc[:,:-1]))
        result_formd.append(get_cosine(ref_formd, X_train_cv[X_train_cv.filename==f].iloc[:,:-1]))
        result_X17as.append(get_cosine(ref_X17as, X_train_cv[X_train_cv.filename==f].iloc[:,:-1]))
        result_TA2.append(get_cosine(ref_TA2, X_train_cv[X_train_cv.filename==f].iloc[:,:-1]))
        
        

    result_df =pd.DataFrame()
    result_df['filename']= X_train_cv['filename']
    result_df['score13f']= result_13f
    result_df['result_82sfc']= result_82sfc
    result_df['result_11k']= result_11k
    result_df['result_formd']= result_formd
    result_df['result_X17as']= result_X17as
    result_df['result_TA2']= result_TA2
    class_dict ={0:'form 13F', 1:'82 SFC', 2:'11K',3:'form D',4:'X 17 AS', 5:'TA 2'}
    classifier_list=[]
    for i in range(result_df.shape[0]):
        li = result_df.iloc[i, 1:].values.tolist()
        max_val = max(li)
        if max_val >0.7:
            classifier_list.append(class_dict[li.index(max_val))
        else:
            classifier_list.append('Others')
    result_df['classifier class'] = classifier_list
    #result_df['classifier class']=['Form' if s >0.7 else 'Others' for s in result_df['score']]
    return result_df
        
  
  
