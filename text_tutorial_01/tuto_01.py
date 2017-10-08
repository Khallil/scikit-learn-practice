from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism','soc.religion.christian', 'comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True,
    random_state=42)

'''Création du CountVectorizer, qui vectorise les mot pour donner la matrice
(id_file, id_word) count_id_word ''' 
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

''' Création du TfidfTransformer, qui converti le CountVectorizer pour obtenir des fréquences
(id_file, id_word) f_apparition'''
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

''' Création du MultinomialNB de Naive_Bayes
 Cet algorithme, va prendre en compte la matrice X_train_tfidf
 pour s'entrainer et connaître en fonction de la fréquence d'apparation des mots
 de quelle catégorie appartient le fichier '''
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

''' Même processus de création cette fois-ci pour le 'test_package' '''
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category])) 
