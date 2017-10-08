from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

categories = ['alt.atheism','soc.religion.christian', 'comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True,
    random_state=42)

docs_new = ['God is love', 'OpenGL on the GPU is fast']

''' Construction de la Pipeline, la Pipeline permet d'enchainer les créations des différents modules'''
text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),])
text_clf.fit(twenty_train.data, twenty_train.target)

second_predicted = text_clf.predict(docs_new)

for doc, category in zip(docs_new, second_predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
''' On fait la prédiction sur le test package'''
predicted = text_clf.predict(docs_test)
''' On compare la liste des prédiction avec le nom des catégories '''
print(np.mean(predicted == twenty_test.target))

