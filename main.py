import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from collections import defaultdict 

df = pd.read_csv('tripadvisor_hotel_reviews.csv')
df = df.head(1000)

df = df[['Review', 'Rating']]
def positive(x_stars):
    return (x_stars > 3)
df['positive'] = df['Rating'].apply(positive)

reviews = df['Review']
sentiment = df['positive']

#implement bag of words algorithm
    
sentences = []
vocab = []
for sent in reviews:
        x = word_tokenize(sent)
        sentence = [w.lower() for w in x if w.isalpha() ]
        sentences.append(sentence)
        for word in sentence:
            if word not in vocab:
                vocab.append(word)
    
index_word = {}
i = 0
for word in vocab:
    index_word[word] = i 
    i += 1

def bag_of_words(sent):
    count_dict = defaultdict(int)
    vec = np.zeros(len(vocab))
    for item in sent:
        count_dict[item] += 1
    for key,item in count_dict.items():
        vec[index_word[key]] = item
    return vec 

##############################

word2vec = []
for sentence in sentences:
     review_vec = bag_of_words(sentence)
     word2vec.append(review_vec)
word2vec = np.array(word2vec)

print("implementing logistic regression here")
w2v_model = LogisticRegression()
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(word2vec, sentiment, test_size=0.2, random_state=101)
w2v_model.fit(X_train_word2vec, y_train_word2vec)

w2v_preds = w2v_model.predict(X_test_word2vec) 
accuracy = accuracy_score(y_test_word2vec, w2v_preds)

print(accuracy)

#with open('tokenizer.pickle', 'wb') as handle:
#   pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)