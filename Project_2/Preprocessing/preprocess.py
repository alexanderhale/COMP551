# basic
import numpy as np
import scipy
import pickle
import pandas as pd
import nltk
import string
import csv
from spellchecker import SpellChecker
from collections import Counter

# natural language toolkit
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag_sents
from nltk.stem import WordNetLemmatizer

# SciKit-Learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# meta-feature extraction
from pymfe.mfe import MFE

# import data
comment_data = pd.read_csv('../Data/reddit_train.csv')
test_data = pd.read_csv('../Data/reddit_test.csv')
print(comment_data.head())
print(test_data.head())

def get_vectorizer(train_data, test_data):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tt.tokenize, stop_words="english", ngram_range=(1,2), min_df=2)
    tfidf_vectorizer.fit(pd.concat([train_data['prep'], test_data['prep']]))
    return tfidf_vectorizer

# helper function for spelling correction
tt = TweetTokenizer()
spell = SpellChecker()
def spellcheck_col(row):
    return " ".join([spell.correction(word) for word in tt.tokenize(row)])

# helper function for lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_col(row):
    return " ".join([lemmatizer.lemmatize(w) for w in tt.tokenize(row)])

# helper function for average word count
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

# stopwords for stopword count
stopwords = nltk.corpus.stopwords.words('english')

# object for meta-feature extraction
mfe = MFE(groups=["general", "statistical", "info-theory"], summary=["min", "median", "max"])

def text_cleanup(data):
    ##### CLEANUP OF INPUT DATA #####
    # punctuation removal
    data['prep'] = data['comments'].str.replace(r'[^\w\s]+', '')

    # lowercase
    data['prep'] = data['prep'].str.lower()

    # convert numbers to 'num'
    data['prep'] = data['prep'].str.replace('(\d+)', ' num ')

    # replace links with 'wasurl'
    data['prep'] = data['prep'].str.replace(r'http(?<=http).*', ' wasurl ')

    # replace newlines and tabs with spaces
    data['prep'] = data['prep'].str.replace(r'\s+', " ")

    # fix any double spaces we created in the previous steps
    data['prep'] = data['prep'].str.replace(" +", " ")

    # typo correction
    data['prep'] = data.prep.apply(spellcheck_col)

    # lemmatization
    data['prep'] = data.prep.apply(lemmatize_col)
    
    return data

def preprocess(inFrame):
    data = inFrame.copy(deep=True)
    
    ##### META-FEATURE EXTRACTION #####
    # word count
    data['word_count'] = data['comments'].apply(lambda x: len(str(x).split(" ")))
    wc = scipy.sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(data['word_count'].to_numpy()))

    # character count
    data['char_count'] = data['comments'].str.len()
    # cc = scipy.sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(data['char_count'].to_numpy()))
    # TODO fix issue that is including NaNs in the char_count list

    # average word length
    data['avg_word'] = data['comments'].apply(lambda x: avg_word(x))
    aw = scipy.sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(data['avg_word'].to_numpy()))

    # stopword count (stopwords will be removed later)
    data['stop_count'] = data['comments'].apply(lambda x: len([x for x in x.split() if x in stopwords]))
    sc = scipy.sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(data['stop_count'].to_numpy()))

    # digit count
    data['digit_count'] = data['comments'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    dc = scipy.sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(data['digit_count'].to_numpy()))
    
    # mathematical meta-feature extraction
    # mfe.fit(comment_data['comments'].tolist(), comment_data['subreddits'].tolist())
    # ft = mfe.extract()
    
    
    ##### PART-OF-SPEECH TAGGING #####
    # data['pos_tag'] = pos_tag_sents(data['prep'].tolist())
    # TODO count totals of each part of speech (noun, adjective, etc) and use the counts as features
    
    
    ##### TF-IDF #####
    tfidf = tfidf_vectorizer.transform(data.prep)
    
    
    ##### FEATURE COMBINATION #####
    feature_matrix = scipy.sparse.hstack((tfidf, wc, aw, sc, dc))
    
    
    ##### FEATURE SELECTION #####
    # TODO if necessary, reduce the number of features by selecting the most informative ones
    
    return feature_matrix

# clean up comments
comment_data = text_cleanup(comment_data)
test_data = text_cleanup(test_data)
print(comment_data.head())
print(test_data.head())

# get tfidf vectorizer
tfidf_vectorizer = get_vectorizer(comment_data, test_data)

# whole training set (for use when making predictions for competition submission)
full_matrix_train = preprocess(comment_data)
print("Full matrix shape:")
print(full_matrix_train.shape)
full_matrix_test = preprocess(test_data)
print("Test matrix shape:")
print(full_matrix_test.shape)

# split up training set (for use when evaluating model accuracies)
X_train = preprocess(comment_data.head(55000))
print("Training matrix shape:")
print(X_train.shape)
X_val = preprocess(comment_data.tail(15000))
print("Validation matrix shape:")
print(X_val.shape)
y_train = comment_data['subreddits'].head(55000)
y_val = comment_data['subreddits'].tail(15000)

# FILE SAVES
# vectorizer
with open('../Data/vectorizer.pk', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

# feature matrices (load on the other side with scipy.sparse.load_npz())
scipy.sparse.save_npz('../Data/feature_matrix_train.npz', full_matrix_train)
scipy.sparse.save_npz('../Data/feature_matrix_test.npz', full_matrix_test)

##### ACCURACY CHECK - TRAIN ON TRAINING SET, VALIDATE ON VALIDATION SET #####
d_tree_val = DecisionTreeClassifier().fit(X_train, y_train)
d_tree_score = d_tree_val.score(X_val, y_val)
print("Decision tree validation score: " + str(d_tree_score))

nb_val = MultinomialNB().fit(X_train, y_train)
nb_score = nb_val.score(X_val, y_val)
print("Naive Bayes validation score: " + str(nb_score))

rf_val = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
rf_score = rf_val.score(X_val, y_val)
print("Random Forest validation score: " + str(rf_score))


##### PREDICTIONS - TRAINING ON FULL TRAINING SET, MAKE PREDICTIONS ON TEST SET #####
d_tree = DecisionTreeClassifier(random_state=0).fit(full_matrix_train, comment_data['subreddits'])
d_tree_predict = d_tree.predict(full_matrix_test)
print("Decision Tree test set predictions:")
print(d_tree_predict)

nb = MultinomialNB().fit(full_matrix_train, comment_data['subreddits'])
nb_predict = nb.predict(full_matrix_test)
print("Naive Bayes test set predictions:")
print(nb_predict)

rf = RandomForestClassifier().fit(full_matrix_train, comment_data['subreddits'])
rf_predict = rf.predict(full_matrix_test)
print("Random Forest test set predictions:")
print(rf_predict)


# check accuracy using same training set values (overfitting, this is just a quick-and-dirty check)
sanity_score = d_tree.score(full_matrix_train, comment_data['subreddits'])
print("Sanity check - decision tree accuracy on training set is: " + str(sanity_score))


# PREDICTION OUTPUTS
with open('predictions.csv', mode='w') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(d_tree_predict)
    writer.writerow(nb_predict)
    writer.writerow(rf_predict)