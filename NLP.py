from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

categories = ['sci.space', 'rec.sport.hockey', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True,
                                  random_state=42)

# print(twenty_train.data[0])

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer(use_idf=False)
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# print(count_vect.vocabulary_.get('laboratory'))
# print(count_vect.transform(['laboratory']))
#
# print(count_vect.vocabulary_.get('WZUM'))
# print(count_vect.transform(['WZUM']))

docs_new = ['There was a new planet discovered',
            'There was a new organ discovered',
            'OpenGL on the GPU is fast']

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

# for doc, category in zip(docs_new, predicted):
    # print('%r => %s' % (doc, twenty_train.target_names[category]))

from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(twenty_train.data, twenty_train.target)

import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,

                              display_labels=clf.classes_)

disp.plot()

plot_confusion_matrix(text_clf, twenty_test.data, twenty_test.target,
                      display_labels=twenty_test.target_names)
plt.show()
