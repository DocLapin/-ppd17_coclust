from coclust.CoclustInfo import CoclustInfo

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.cluster import normalized_mutual_info_score

categories = [
    'rec.motorcycles',
    'rec.sport.baseball',
    'comp.graphics',
    'sci.space',
    'talk.politics.mideast'
]

ng5 = fetch_20newsgroups(categories=categories, shuffle=True)

true_labels = ng5.target

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('coclust', CoclustInfo()),
])

pipeline.set_params(coclust__n_clusters=5)
pipeline.fit(ng5.data)

predicted_labels = pipeline.named_steps['coclust'].row_labels_

nmi = normalized_mutual_info_score(true_labels, predicted_labels)

print(nmi)