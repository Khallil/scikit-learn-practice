from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

categories = ['alt.atheism','soc.religion.christ', 'compt.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True,
    random_state=42)

