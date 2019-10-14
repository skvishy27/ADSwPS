#synthetic datasets
# Linear support Vector Machines

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_C2, y_C2 = make_classification(n_samples = 100, n_features = 2,
                                 n_redundant =0, n_informative = 2,
                                 n_clusters_per_class =1, flip_y=0.1,
                                 class_sep=0.5,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_c2, y_C2, random_state =0)

clf = SVC(kernel = 'linear', C=1.0).fit(X_train, y_train)

print('Accuracy of LSVM classifier of training set : {:.2f}'
      .format(clf.score(X_train,y_train)))
print('Accuracy of LSVM classifier of testing set : {:.2f}'
      .format(clf.score(X_test, y_test)))


# Application to real dataset

from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state =0)

clf = LinearSVC().fit(X_train, y_train)

print('Accuracy of Linear SVC classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC Classifier on test set:{:.2f}'
      .format(clf.score(X_test, y_test)))
