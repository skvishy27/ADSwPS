# Naive Bayes Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification,make_blobs
from sklearn.model_selection import train_test_split

# synthetic dataset for classification (binary)
X_C2, y_C2 = make_classification(n_samples = 100, n_features = 2,
                                n_redundant = 0,n_informative = 2,
                                n_clusters_per_class = 1, flip_y = 0.1,
                                class_sep = 0.5, random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
nbclf = GaussianNB().fit(X_train, y_train)

train_score = nbclf.score(X_train, y_train)
test_score = nbclf.score(X_test, y_test)
print('Train Score: {:.2f}\nTest score: {:.2f}'
      .format(train_score,test_score))



# more difficult synthetic dataset for classification (binary)
# with classes that are not linearly separable


X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2,
                                                   random_state = 0)
nbclf = GaussianNB().fit(X_train, y_train)
train_score = nbclf.score(X_train, y_train)
test_score = nbclf.score(X_test, y_test)
print('Train Score: {:.2f}\nTest Score: {:.2f}'
     .format(train_score, test_score))



# Application to a real- world dataset

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
nbclf = GaussianNB().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of GaussianNB classifier on training set: {:.2f}'
     .format(nbclf.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'
     .format(nbclf.score(X_test, y_test)))

